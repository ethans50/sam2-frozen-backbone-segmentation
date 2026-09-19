"""Map CamVid11 and Cityscapes trainId to the official Cityscapes 7-category space
(flat/construction/object/nature/sky/human/vehicle) for Phase 2 cross-dataset eval.

category definitions come from cityscapesscripts' labels.py (not hardcoded). CamVid11 ->
category assignment is derived by hand against that definition (see DETAILS.md "Phase 2").
CamVid's Bicyclist has no clean match (Cityscapes splits rider/bicycle) so it's ignored
in cross-eval instead of forced into either category.
"""
import json
from pathlib import Path

IGNORE_INDEX = 255

CAMVID11_TO_CATEGORY = {
    "Sky": "sky",
    "Building": "construction",
    "Pole": "object",
    "Road": "flat",
    "Pavement": "flat",
    "Tree": "nature",
    "SignSymbol": "object",
    "Fence": "construction",
    "Car": "vehicle",
    "Pedestrian": "human",
    "Bicyclist": None,  # ignored in cross-eval, see DETAILS.md "Bicyclist 예외 처리 근거"
}


def main():
    from cityscapesscripts.helpers.labels import labels as cs_labels

    categories = sorted({l.category for l in cs_labels if l.category != "void"})
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Cityscapes trainId (0-18) -> category index
    cs_trainid_to_cat = [IGNORE_INDEX] * 19
    for l in cs_labels:
        if l.trainId not in range(19):
            continue
        cs_trainid_to_cat[l.trainId] = cat_to_idx[l.category]

    # CamVid11 (0-10) -> category index (must match build_class_map.py class order)
    with open("data/camvid/class_map.json") as f:
        camvid_meta = json.load(f)
    camvid_classes = camvid_meta["classes"]

    camvid_to_cat = []
    for name in camvid_classes:
        cat_name = CAMVID11_TO_CATEGORY[name]
        camvid_to_cat.append(IGNORE_INDEX if cat_name is None else cat_to_idx[cat_name])

    meta = {
        "categories": categories,
        "ignore_index": IGNORE_INDEX,
        "cityscapes_trainid_to_category": cs_trainid_to_cat,
        "camvid_class_to_category": camvid_to_cat,
        "camvid_classes_order": camvid_classes,
    }
    out_path = Path("data/category_map.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"common categories ({len(categories)}): {categories}")
    print(f"Cityscapes trainId -> category: {cs_trainid_to_cat}")
    print(f"CamVid11 -> category: {dict(zip(camvid_classes, camvid_to_cat))}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
