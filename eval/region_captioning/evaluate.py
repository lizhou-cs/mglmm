import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')))

import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from eval.utils.utils import save_metrics_to_json


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - Region Captioning")

    parser.add_argument("--annotation_file",
                        default="data/RefCoco_Reg/mdetr_annotations/finetune_refcocog_val_captions.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to save the results.")

    return parser.parse_args()


def calculate_metrics(annotation_file, results_dir):
    # Load the annotation file
    ignore_files = ["merged.json", "metrics.json"]
    coco = COCO(annotation_file)

    # Merge and load the results files
    merged_file_path = f"{results_dir}/merged.json"

    all_results = []
    for result_file in os.listdir(results_dir):
        if result_file.endswith(".json") and result_file not in ignore_files:
            all_results += json.load(open(f"{results_dir}/{result_file}", "r"))
    
    image_id_set = set()
    merged_results = []

    for result in all_results:
        if result["image_id"] in image_id_set:
            print(f"Image {result['image_id']} has multiple results.")
        else:
            image_id_set.add(result["image_id"])
            merged_results.append(result)

    with open(merged_file_path, 'w') as f:
        json.dump(merged_results, f)
    
    coco_result = coco.loadRes(merged_file_path)

    # Create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate results
    coco_eval.params['image_id'] = coco_result.getImgIds()
    try:
        coco_eval.evaluate()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in evaluation: {e}")

    # Print and save the output evaluation scores
    output_file_path = f"{results_dir}/metrics.json"
    all_metrics = {}
    for metric, score in coco_eval.eval.items():
        # save score with .3f precision
        all_metrics[metric] = round(score, 3)
    save_metrics_to_json(all_metrics, output_file_path)
    print(all_metrics)

    # remove old results files
    for result_file in os.listdir(results_dir):
        if result_file.endswith(".json") and result_file not in ignore_files:
            os.remove(f"{results_dir}/{result_file}")


if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.annotation_file, args.results_dir)
