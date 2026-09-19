# Design Process and Rationale

이 문서는 이 레포지토리의 설계 결정(architecture, hyperparameter, evaluation protocol)이 왜 지금 형태가
되었는지에 대한 근거와, 각 결정을 뒷받침하는 실측값을 기록한다. 결과 자체는 [`README.kr.md`](README.kr.md)에
있으며, 이 문서는 그 결과가 어떤 과정을 거쳐 나왔는지를 설명한다.

## Contents

- [Research Question](#research-question)
- [Design Principles](#design-principles)
- [Phase 1 Configuration: CamVid](#phase-1-configuration-camvid)
- [Phase 2 Configuration: Cityscapes and Cross-Dataset Evaluation](#phase-2-configuration-cityscapes-and-cross-dataset-evaluation)
- [Qualitative Error Analysis](#qualitative-error-analysis)
- [References](#references)

## Research Question

자연 이미지로 pretrained된 SAM2의 image encoder를 전혀 수정하지 않고, 그 위에 경량 decoder만 새로
학습시켜 표준적인 다중 클래스 semantic segmentation에 얼마나 잘 transfer되는가? 그리고 한 dataset으로
학습한 decoder가 다른 domain의 dataset에도 일반화되는가, 아니면 학습에 쓰인 dataset에만 overfitting된
얕은 mapping을 학습한 것인가? 두 번째 질문은 cross-dataset evaluation으로 답한다: Cityscapes로만 학습한
decoder를 추가 학습 없이 CamVid — 카메라, 도시, class taxonomy가 전부 다른 dataset — 에 그대로 평가한다.

## Design Principles

### Why SAM2

SAM2는 checkpoint, 코드, 문서 등 생태계가 안정적이고, `tiny`부터 `large`까지 공식 checkpoint가 여러
크기로 배포되어 있어 encoder 크기에 따른 transfer 성능 비교가 용이하다. 이런 종류의 frozen-backbone
transfer 실험에 적합한 표준적인 선택이다.

### Why Freeze the Encoder

1. 대형 pretrained 모델을 작거나 중간 규모의 dataset에 통째로 fine-tuning하면 overfitting과
   catastrophic forgetting 위험이 크다 — encoder를 freeze하는 것이 transfer learning에서 이를 피하는
   표준적인 방법이다.
2. encoder를 `torch.no_grad()`로 돌리면 backpropagation을 위한 activation을 저장하지 않아도 되므로,
   완전히 학습 가능한 모델보다 더 큰 batch size를 쓸 수 있다.
3. decoder만 학습하기 때문에, cross-dataset evaluation이 순수한 질문 하나를 측정한다: encoder 자체가
   dataset 고유의 특징을 암기했는지가 아니라, decoder가 encoder의 범용 representation 위에서 얼마나
   잘 일반화되는지. 이것이 Phase 2가 답하고자 하는 핵심 질문이다.

### Decoder Design

decoder는 SAM2의 `backbone_fpn`(multi-scale feature map 리스트)을 입력으로 받는다:

1. 각 scale을 1x1 convolution으로 공통 채널 수로 projection한다.
2. 가장 해상도가 높은 scale의 해상도 기준으로 나머지 scale을 bilinear upsample한다.
3. upsample된 feature들을 concatenate한 뒤 3x3 convolution 두 층을 통과시킨다.
4. 마지막 1x1 convolution으로 `num_classes` 채널을 출력한다.
5. loss 계산 전에 logit을 입력 해상도로 bilinear upsample한다.

decoder에는 비대칭적이거나 window 기반의 attention 메커니즘을 쓰지 않는다 — 입력 이미지가 표준적인
자연 이미지 종횡비이고 이용할 만한 특수 구조가 없으므로, 추가적인 architecture 복잡도를 정당화할
근거가 없다.

### Loss and Evaluation Design

- **Loss**: CrossEntropy + Dice 결합. 두 dataset 모두에 존재하는 class 불균형(road/sky 같은 큰 class
  대 pole/pedestrian 같은 작은 class) 하에서 Dice loss가 CrossEntropy 단독보다 더 강건하다.
- **Evaluation**: 표준 mIoU와 별도로 class별 IoU도 함께 보고한다 — 어떤 class가 약한지 확인하는 것
  자체가 research question의 일부이지, 부수적인 사항이 아니다.
- **ignore index(void class)는 loss와 mIoU 계산 양쪽에서 반드시 제외한다.** 어느 한쪽에서라도 빠뜨리면
  보고되는 지표가 왜곡된다.

## Phase 1 Configuration: CamVid

| 항목 | 값 | 근거 |
|---|---|---|
| Dataset | CamVid | — |
| Class taxonomy | CamVid11: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist (+ Void = ignore). RGB→index mapping은 `scripts/build_class_map.py`가 공식 `data/camvid/raw/class_dict.csv`(32-class 원본)에서 직접 생성 | 32→11 grouping 규칙은 MathWorks "Semantic Segmentation Using Deep Learning" 참조 구현(`camvidPixelLabelIDs()`, SegNet 방식 grouping)을 따름. `class_dict.csv`의 31개 non-Void class와 1:1로 정확히 매칭됨을 확인(누락/중복 없음), 6장의 label 이미지로 시각적 대조까지 완료 |
| Ignore index (Void) | `255` | `build_class_map.py`에서 지정, loss와 mIoU 양쪽에서 제외 |
| Data split | `GeorgeSeif/Semantic-Segmentation-Suite` repository가 배포하는 train(421)/val(112)/test(168) split | 외부에서 검증 가능하도록 재분할 없이 배포된 그대로 사용 |
| SAM2 checkpoint | `sam2.1-hiera-tiny` | pilot phase의 초기 선택 |
| Backbone 처리 | encoder 완전 freeze, gradient 없음(`torch.no_grad()`) | — |
| Decoder architecture | multi-scale projection(1x1 conv, `proj_dim=128`) → 최고해상도 scale 기준 bilinear upsample → concat → 3x3 conv 2층(`hidden_dim=128`, ReLU) → 1x1 conv(`num_classes`) → 입력 해상도로 bilinear upsample. 구현: `scripts/model.py`의 `Decoder`/`SegModel` | — |
| 실측된 `backbone_fpn` shape | `SegModel`은 `sam2_model.image_encoder`(FPN neck 출력)를 직접 호출하며, `sam_mask_decoder.conv_s0/conv_s1`(SAM 자체의 prompt-decoder 경로, level 0/1을 32/64 채널로 축소하는 부분)은 거치지 않는다 — 우리 pipeline엔 불필요. 512x512 입력 기준 3개 scale 전부 채널 256, 해상도 128x128/64x64/32x32 (stride 4/8/16) | `python scripts/model.py`로 직접 실측(가정하지 않음) |
| 입력 해상도 | **512x512** | batch size 4 기준 1024 대비 forward 6배 빠름(0.42s→0.07s), peak VRAM도 낮음(1.57GB→1.05GB). encoder는 임의 해상도를 지원(Hiera의 windowed positional embedding이 내부적으로 보간됨, 512/1024 양쪽 다 에러 없이 동작) |
| Loss | CrossEntropy + Dice, 1:1 가중치 | — |
| Optimizer | AdamW | — |
| Learning rate | `1e-3` | 권장 범위(1e-3~3e-4) 중 처음 시도한 값으로 바로 수렴 성공 |
| Epochs | 40 (early stopping patience=8로 설정했으나 발동 안 함 — val mIoU가 40 epoch 내내 느리게나마 계속 개선됨) | 최종 val mIoU **0.7288** @ epoch 40 |
| Batch size | **24** | 합성 벤치마크상으로는 32가 안전해 보였으나(peak 4.47GB/8GB), 실제 학습 루프(dataloader+augmentation 포함)에서 32 사용 시 CUDA allocator retry 경고가 1회 발생 — 더 안전한 24로 낮춤 |
| Evaluation metric | mIoU + class별 IoU (torchmetrics, ignore index 제외) | — |

## Phase 2 Configuration: Cityscapes and Cross-Dataset Evaluation

### Objective

Cityscapes(50개 도시, 수천 장)로 학습 데이터 규모와 class별 표본 수를 크게 늘리는 한편, **CamVid를
decoder 학습에서 완전히 배제하고 held-out evaluation set으로만 사용**한다. 이는 단일 dataset 내부의
일반적인 train/val/test split보다 훨씬 엄격한 일반화 검증이다. 더 오래/많이 학습하는 것만으로도
in-domain 지표는 overfitting을 통해 얼마든지 오를 수 있으므로, 그 in-domain 수치를 주된 성공 기준으로
삼지 않는다. 주된 기준은 **"Cityscapes로만 학습한 decoder가, 한 번도 보지 못한 CamVid 이미지에서
category 단위로 얼마나 잘 맞는가"**이다.

### Dataset: Cityscapes

공식 배포: fine annotation 기준 train 2,975장 + val 500장 (test 1,525장은 공개 label이 없고 공식
evaluation server 제출 전용이라 여기서는 쓰지 않음 — 대신 val을 model selection에 사용하고, 진짜
일반화 주장은 아래 CamVid cross-evaluation에서 나온다). Label은 `gtFine/*_labelIds.png`(34-class
`id`)에서 19-class `trainId`로 디코딩하는데, `cityscapesscripts` 패키지의 공식 `labels` 테이블을
코드에서 직접 import해서 사용한다(직접 옮겨 적지 않음 — mapping 전사 오류 가능성 자체를 제거).

### Class Taxonomy: Cityscapes 19-Class (Training) and 7-Category (Cross-Evaluation)

출처: `cityscapesscripts/cityscapesscripts/helpers/labels.py`
([cityscapesScripts](https://github.com/mcordts/cityscapesScripts)), 직접 import(재입력 없음):

| id | name | trainId | category |
|---|---|---|---|
| 0-6 | unlabeled / ego vehicle / rectification border / out of roi / static / dynamic / ground | 255 (ignore) | void |
| 7 | road | 0 | flat |
| 8 | sidewalk | 1 | flat |
| 9 | parking | 255 | flat |
| 10 | rail track | 255 | flat |
| 11 | building | 2 | construction |
| 12 | wall | 3 | construction |
| 13 | fence | 4 | construction |
| 14 | guard rail | 255 | construction |
| 15 | bridge | 255 | construction |
| 16 | tunnel | 255 | construction |
| 17 | pole | 5 | object |
| 18 | polegroup | 255 | object |
| 19 | traffic light | 6 | object |
| 20 | traffic sign | 7 | object |
| 21 | vegetation | 8 | nature |
| 22 | terrain | 9 | nature |
| 23 | sky | 10 | sky |
| 24 | person | 11 | human |
| 25 | rider | 12 | human |
| 26 | car | 13 | vehicle |
| 27 | truck | 14 | vehicle |
| 28 | bus | 15 | vehicle |
| 29 | caravan | 255 | vehicle |
| 30 | trailer | 255 | vehicle |
| 31 | train | 16 | vehicle |
| 32 | motorcycle | 17 | vehicle |
| 33 | bicycle | 18 | vehicle |
| -1 | license plate | 255 | vehicle |

decoder는 표준 Cityscapes 관례대로 19-class `trainId`로 학습하며, 이를 통해 문헌과 직접 비교 가능한
Cityscapes val mIoU도 부가 지표로 보고한다. 7-category grouping(flat/construction/object/nature/sky/
human/vehicle)은 이 프로젝트가 새로 만든 것이 아니라 Cityscapes 공식 `category` 정의를 그대로 재사용한
것이며, cross-dataset evaluation의 공통 label space로 쓰인다:

| CamVid11 class | → category |
|---|---|
| Sky | sky |
| Building | construction |
| Pole | object |
| Road | flat |
| Pavement | flat |
| Tree | nature |
| SignSymbol | object |
| Fence | construction |
| Car | vehicle |
| Pedestrian | human |
| **Bicyclist** | **제외 (cross-evaluation에서 ignore)** |

**Bicyclist 예외 처리 근거**: CamVid의 `Bicyclist`는 "자전거를 탄 사람"을 사람과 자전거의 구분 없이
하나의 class로 라벨링한다. Cityscapes는 이를 `rider`(human category)와 `bicycle`(vehicle category)로
분리한다. 둘 중 하나로 강제 편입시키면 근거 없는 임의적 결정이 되므로, `Bicyclist` 픽셀은
cross-evaluation mIoU 계산에서 제외(ignore)하고, 이 사실을 모든 결과 보고에서 명시적으로 밝힌다.

### Model Selection

Phase 1은 `sam2.1-hiera-tiny`를 사용했다. Phase 2는 학습 데이터가 훨씬 많아져, 8GB VRAM 안에서
가능하다면 더 큰 encoder의 표현력을 활용할 가치가 있다. 선택 전에 SAM2의 네 가지 크기를 전부
직접 benchmark했다(forward pass만, encoder는 `torch.no_grad()`, 목표 해상도/batch size 기준):

| Checkpoint | Peak VRAM (forward only) |
|---|---|
| tiny | 0.47GB |
| small | 1.20GB |
| base_plus | 1.37GB |
| large | 1.99GB |

네 가지 전부 8GB 안에 여유 있게 들어간다 — encoder가 `no_grad()`로 동작해 backpropagation용
activation을 보관하지 않으므로, 가장 큰 checkpoint조차 메모리 부담이 크지 않다. Phase 2의 더 큰
dataset을 충분히 활용하기 위해 `large`를 선택했다.

### Resolution

Cityscapes 이미지는 기본적으로 2048x1024(2:1 종횡비)로, CamVid의 4:3과 다르다. SAM2 encoder가
정사각형이 아닌 입력(정사각형으로 resize하면 이미지가 왜곡됨)을 올바르게 처리하는지를 가정하지 않고
직접 측정했다: 1024x1024, 1024x512, 768x384, 512x256, 512x1024 전부 에러 없이 동작했다 — Hiera의
windowed attention이 입력 크기에 맞춰 내부적으로 적응하기 때문이다. **1024x512**를 선택했다 — 원본
해상도의 절반, 종횡비 보존, 왜곡과 padding 모두 불필요.

### Training and Evaluation Strategy

더 오래/많이 학습해서 in-domain 지표를 올리는 것 자체는 일반화의 증거가 아니다 — overfitting의
증거일 수도 있다. cross-dataset 결과가 의미를 가지려면 다음 네 가지 제약을 지켰다:

1. **Model selection(best checkpoint)은 Cityscapes validation mIoU만으로 한다.** checkpoint 선택 시
   CamVid는 절대 참고하지 않는다.
2. **CamVid는 마지막에 cross-dataset evaluation 용도로 단 한 번만 사용한다**(Phase 1에서 이미 검증된
   label index를 재사용) — 학습 데이터에 섞지 않는다.
3. Cityscapes val mIoU는 계속 오르는데 CamVid cross-evaluation mIoU가 정체되거나 하락한다면, 이는
   Cityscapes domain에 대한 overfitting의 증거로 명시적으로 기록한다 — 이 실험이 바로 탐지하고자 하는
   failure mode다.
4. Epoch 수는 Cityscapes val mIoU 기준 early stopping을 쓰되, Phase 1보다 patience를 넉넉히 잡아
   자연스러운 수렴 지점을 찾는다(validation set 자체에 overfitting될 정도로 길게 돌리지 않도록 주의).

### Confirmed Configuration

| 항목 | 값 | 근거 |
|---|---|---|
| 학습 데이터 | Cityscapes fine (train 2,975 + val 500) | 다운로드/디코딩 완료, archive checksum과 공식 split과 정확히 일치하는 최종 파일 개수로 무결성 검증 |
| 학습 class taxonomy | Cityscapes 19-class `trainId` | `scripts/build_class_map_cityscapes.py`에서 공식 `cityscapesscripts` 패키지를 설치해 `labels` 테이블을 직접 import(수동 전사 없음) |
| Cross-evaluation taxonomy | 7-category(flat/construction/object/nature/sky/human/vehicle), Bicyclist는 ignore | `scripts/build_category_map.py` → `data/category_map.json` |
| SAM2 checkpoint | **`sam2.1-hiera-large`** | Model Selection 절 참고 |
| 입력 해상도 | **1024(W) x 512(H)** | Resolution 절 참고 |
| Batch size | **8** | `large` checkpoint로 forward+backward+optimizer step 전체를 실측: batch 8은 peak 3.49GB(안전), batch 16은 5.88GB(성공하지만 여유 적음), batch 24는 OOM. Phase 1에서 합성 벤치마크와 실제 pipeline VRAM 사용량 차이를 겪었기 때문에 처음부터 여유 있는 값을 선택 |
| Epoch / early stopping | patience=12, 최대 60 epoch | epoch 52에서 조기 종료, best checkpoint는 **epoch 40** (`checkpoints/best_decoder_cityscapes.pt`) |
| 결과: Cityscapes val (19-class) | **mIoU 0.6437** | 크고 흔한 class가 가장 높음(road 0.97, sky 0.94, car 0.91), 작고 드문 class가 가장 낮음(train 0.29, bus 0.41, motorcycle 0.41, rider 0.42) — Phase 1과 동일한 패턴 |
| 결과: CamVid cross-evaluation (7-category, 전체 701장) | **mIoU 0.7470** | Cityscapes val 수치보다 높지만 직접 비교 대상은 아님(7-category vs 19-class). 다만 domain 붕괴 신호는 전혀 없음. category별로 flat 0.98, sky 0.86, construction 0.86이 높고, object(가는 물체) 0.45가 가장 낮음 — Phase 1과 동일한 failure mode 재현 |

## Qualitative Error Analysis

`eval.py`/`eval_phase2.py`가 저장하는 대표 샘플 6장만으로는 잘 된 case와 안 된 case를 골고루 확인하기에
부족했다(Phase 1의 6장은 우연히 전부 같은 sequence의 연속 frame이었다). 각 held-out set 전체를 이미지
단위로 채점해 상위/하위 5장을 뽑기 위해 `scripts/visualize_cases.py`를 추가했다:

| 파일 접두어 | 대상 | 크기 | 이미지별 점수 범위 |
|---|---|---|---|
| `eval_test_best{1-5}_*.png` / `eval_test_worst{1-5}_*.png` | Phase 1, CamVid test (11-class) | 168장 | 0.4939 - 0.8348 |
| `cross_eval_best{1-5}_*.png` / `cross_eval_worst{1-5}_*.png` | Phase 2, CamVid cross-evaluation (7-category) | 701장 | 0.3944 - 0.8404 |

**이 이미지별 점수가 `results_phase1.json`/`results_phase2.json`의 공식 mIoU와 다른 두 가지 이유(둘 다 공식
지표에는 영향 없음):**

1. **계산 범위**: 공식 지표는 held-out set 전체 픽셀의 confusion matrix를 한 번에 누적한 뒤
   (`torchmetrics.MulticlassJaccardIndex(average="macro")`) class별 IoU를 평균한다. 여기서의 순위용
   점수는 이미지 한 장 단위로 confusion matrix를 계산하고, 그 이미지의 ground truth에 실제로 등장하는
   class만 평균한다(등장하지 않는 class를 집계에 포함하면, 예를 들어 Bicyclist가 없는 frame이 IoU 0으로
   잡혀 부당하게 낮아진다).
2. **역할 분리**: `visualize_cases.py`는 `results*.json`을 전혀 읽거나 쓰지 않는다 — 순수하게 시각화용
   이미지를 고르기 위한 보조 점수이며, 공식 evaluation script와 완전히 분리된 코드 경로다. 이 순위 점수가
   공식 지표에 영향을 줄 방법이 없다.

**worst case에서 관찰되는 패턴:**

- **가늘고 반복적인 구조물(가로등 기둥, 잎이 없는 나뭇가지, bollard)이 뭉뚝한 덩어리로 뭉개지거나
  누락된다.** 구조적인 원인이다: decoder가 예측을 만드는 실제 해상도는 SAM2의 가장 고해상도
  feature scale(stride 4, 512 입력 기준 128x128 — Decoder Design 및 Phase 1 표 참고)뿐이고, 거기서
  바로 4배로 bilinear upsample한다. 그 격자 크기보다 가는 구조는 마지막 upsample 이전에 이미 사라진다
  (`eval_test_worst1/2/4`, `cross_eval_worst4`에서 확인).
- **작은 물체가 여러 개 밀집한 복잡한 장면일수록 점수가 낮다.** 인접한 class 경계끼리 서로 스며들며
  (예: `cross_eval_worst4`에서 vehicle/human/object가 뒤섞임) 위 해상도 한계를 더 두드러지게 만든다.
- **Phase 2 cross-evaluation에서만 나타나는 패턴**: worst 5장 중 2장(`cross_eval_worst1`, `worst2`,
  같은 주유소 장면의 연속 frame)에서 그림자 진 구조물 아래 빈 공간에 실제로 존재하지 않는 "vehicle"을
  예측하는 false positive가 나타났다. 이는 해상도 한계가 아니라 domain shift로 보인다 — 모델이
  Cityscapes 도로 장면에서 "지붕/구조물 아래의 어두운 덩어리 = vehicle"이라는 패턴을 학습했고, 이를
  CamVid의 낯선 장면 구도에 과도하게 일반화해 적용한 것으로 해석된다. Phase 1에서 이미 알려진
  가는 구조물 한계와는 성격이 다르며, cross-dataset evaluation을 통해서만 드러난 failure mode다.

## References

- CamVid: Cambridge-driving Labeled Video Database.
- CamVid11 class taxonomy: SegNet / Bayesian SegNet 계열 문헌에서 널리 쓰이는 표준 축소 class set.
- SAM2: <https://github.com/facebookresearch/sam2>
- Cityscapes: <https://www.cityscapes-dataset.com/>,
  <https://github.com/mcordts/cityscapesScripts>
