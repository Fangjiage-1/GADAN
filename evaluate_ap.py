"""Evaluate a GADAN checkpoint with AP and IoU-based grounding metrics."""

import torch_patch
import argparse
import random
import time

import numpy as np
import torch

import os
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import opts
import util.misc as utils
from datasets import build_dataset
from models.gadan import build as build_model
from tools.colormap import colormap
from util.checkpoint import load_model_state_strict, restore_architecture_args
from util.misc import AverageMeter


color_list = colormap()
color_list = color_list.astype('uint8').tolist()

Visualize_bbox = False
save_visualize_path_prefix = "test_output"
version = "test_ap_bilstm_gsbi"


def main(args):
    args.masks = False
    print("Inference only supports for batch size = 1")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.visualize and not os.path.exists(save_visualize_path_prefix):
        os.makedirs(save_visualize_path_prefix)

    if not args.resume:
        raise ValueError('Please specify the checkpoint for inference.')

    checkpoint = torch.load(args.resume, map_location='cpu')
    restore_architecture_args(args, checkpoint)

    test_dataset = build_dataset(args.dataset_file, image_set='test', args=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=4)

    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    load_model_state_strict(model, checkpoint)

    evaluate(test_loader, model, args)


def evaluate(test_loader, model, args):
    batch_time = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    device = args.device
    model.eval()
    end = time.time()

    img_list = []
    count = 0
    all_predictions = []
    all_gts = {}
    total_gts = 0
    for batch_idx, (img, targets, dw, dh, img_path, ratio) in enumerate(test_loader):
        h_resize, w_resize = img.shape[-2:]
        img = img.to(device)
        captions = targets["caption"]
        size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
        target = {"size": size}

        with torch.no_grad():
            outputs = model(img, captions, [target])

        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid()
        pred_score = pred_score.squeeze(0)
        max_score, _ = pred_score.max(-1)
        _, max_ind = max_score.max(-1)
        pred_bbox = pred_bbox[0, max_ind]

        pred_bbox = rescale_bboxes(pred_bbox.detach(), (w_resize, h_resize)).numpy()
        target_bbox = rescale_bboxes(targets["boxes"].squeeze(), (w_resize, h_resize)).numpy()

        pred_bbox[0], pred_bbox[2] = (pred_bbox[0] - dw) / ratio, (pred_bbox[2] - dw) / ratio
        pred_bbox[1], pred_bbox[3] = (pred_bbox[1] - dh) / ratio, (pred_bbox[3] - dh) / ratio
        target_bbox[0], target_bbox[2] = (target_bbox[0] - dw) / ratio, (target_bbox[2] - dw) / ratio
        target_bbox[1], target_bbox[3] = (target_bbox[1] - dh) / ratio, (target_bbox[3] - dh) / ratio

        if Visualize_bbox:
            source_img = Image.open(img_path[0]).convert('RGB')
            draw = ImageDraw.Draw(source_img)
            draw_boxes = pred_bbox.tolist()
            xmin, ymin, xmax, ymax = draw_boxes[0:4]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[9]), width=2)
            save_visualize_path_dir = os.path.join(save_visualize_path_prefix, version)
            if not os.path.exists(save_visualize_path_dir):
                os.makedirs(save_visualize_path_dir)
            img_name = img_path[0].split('/')[-1]
            if img_name not in img_list:
                img_list.append(img_name)
            else:
                count += 1
                img_name = str(count) + '_' + img_name
            save_visualize_path = os.path.join(save_visualize_path_dir, img_name)
            source_img.save(save_visualize_path)

        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox)
        cumInterArea = np.sum(np.array(interArea.data.numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.numpy()))
        accu5 = np.sum(np.array((iou.data.numpy() > 0.5), dtype=float)) / 1
        accu6 = np.sum(np.array((iou.data.numpy() > 0.6), dtype=float)) / 1
        accu7 = np.sum(np.array((iou.data.numpy() > 0.7), dtype=float)) / 1
        accu8 = np.sum(np.array((iou.data.numpy() > 0.8), dtype=float)) / 1
        accu9 = np.sum(np.array((iou.data.numpy() > 0.9), dtype=float)) / 1

        meanIoU.update(torch.mean(iou).item(), img.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)

        acc5.update(accu5, img.size(0))
        acc6.update(accu6, img.size(0))
        acc7.update(accu7, img.size(0))
        acc8.update(accu8, img.size(0))
        acc9.update(accu9, img.size(0))

        try:
            conf_score = float(max_score[max_ind].item())
        except Exception:
            conf_score = float(max_score.item()) if hasattr(max_score, 'item') else float(max_score)

        all_predictions.append({
            'sample_id': batch_idx,
            'image_id': batch_idx,
            'expression': captions[0] if isinstance(captions, (list, tuple)) else str(captions),
            'bbox': [float(pred_bbox[0]), float(pred_bbox[1]), float(pred_bbox[2]), float(pred_bbox[3])],
            'gt_box': [float(target_bbox[0]), float(target_bbox[1]), float(target_bbox[2]), float(target_bbox[3])],
            'score': conf_score,
            'iou': float(iou.item()) if hasattr(iou, 'item') else float(iou),
        })
        gt_box_list = all_gts.get(batch_idx, [])
        gt_box_list.append([float(target_bbox[0]), float(target_bbox[1]), float(target_bbox[2]), float(target_bbox[3])])
        all_gts[batch_idx] = gt_box_list
        total_gts += 1

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.avg:.3f}\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t' \
                        'acc@0.7: {acc7.avg:.4f}\t' \
                        'acc@0.8: {acc8.avg:.4f}\t' \
                        'acc@0.9: {acc9.avg:.4f}\t' \
                        'meanIoU: {meanIoU.avg:.4f}\t' \
                        'cumuIoU: {cumuIoU:.4f}\t' \
                .format(
                batch_idx, len(test_loader), batch_time=batch_time,
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9,
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
            print(print_str)
    final_str = 'acc@0.5: {acc5.avg:.4f}\t' 'acc@0.6: {acc6.avg:.4f}\t' 'acc@0.7: {acc7.avg:.4f}\t' \
                'acc@0.8: {acc8.avg:.4f}\t' 'acc@0.9: {acc9.avg:.4f}\t' \
                'meanIoU: {meanIoU.avg:.4f}\t' 'cumuIoU: {cumuIoU:.4f}\t' \
        .format(acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9,
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
    print(final_str)
    print(version)

    def compute_iou_xyxy(boxA, boxB):
        xa1, ya1, xa2, ya2 = boxA
        xb1, yb1, xb2, yb2 = boxB
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def compute_ap_from_matches(preds, gts, iou_thresh=0.5):
        if len(gts) == 0:
            return 0.0
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        image_gt_matched = {img: [False] * len(boxes) for img, boxes in gts.items()}
        tp = []
        fp = []
        for pred in preds_sorted:
            img_id = pred['image_id']
            pred_box = pred['bbox']
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts.get(img_id, [])):
                iou = compute_iou_xyxy(pred_box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_thresh and best_idx >= 0 and (not image_gt_matched.get(img_id, [False])[best_idx]):
                tp.append(1)
                fp.append(0)
                image_gt_matched[img_id][best_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp).astype(np.float32)
        fp_cum = np.cumsum(fp).astype(np.float32)
        recalls = tp_cum / float(sum(len(v) for v in gts.values()))
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float32).eps)
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = 0.0
        for i in indices:
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
        return ap

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    ap_list = []
    for threshold in iou_thresholds:
        ap_t = compute_ap_from_matches(all_predictions, all_gts, iou_thresh=float(threshold))
        ap_list.append(ap_t)
        print(f"AP@{threshold:.2f}: {ap_t:.4f}")
    mean_ap = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
    print(f"mAP (IoU=0.5:0.95 step 0.05): {mean_ap:.4f}")
    print(f"Total GTs: {total_gts}, Total predictions: {len(all_predictions)}")

    # Save per-sample predictions to JSON if requested
    if hasattr(args, 'save_json') and args.save_json:
        import json as _json
        # Convert bbox -> pred_box for compatibility with analysis script
        save_data = []
        for p in all_predictions:
            save_data.append({
                'sample_id': p['sample_id'],
                'image_id': p['image_id'],
                'expression': p['expression'],
                'pred_box': p['bbox'],
                'gt_box': p['gt_box'],
                'score': p['score'],
                'iou': p['iou'],
            })
        save_path = args.save_json
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            _json.dump(save_data, f, indent=2)
        print(f'Predictions saved to: {save_path}')


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(box1[0]), torch.tensor(box1[1]), torch.tensor(box1[2]), torch.tensor(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(box2[0]), torch.tensor(box2[1]), torch.tensor(box2[2]), torch.tensor(box2[3])
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    return (inter_area + 1e-6) / (union_area + 1e-6), inter_area, union_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=0)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_bilstm_gsbi_parser():
    parser = argparse.ArgumentParser('Refer_RSVG inference script (with mAP)', parents=[opts.get_args_parser()])
    parser.add_argument('--text_encoder_type', default='bilstm', type=str,
                        choices=['bilstm', 'roberta', 'roberta_frozen', 'bert_tiny'],
                        help='Text encoder type: bilstm, roberta, roberta_frozen, bert_tiny')
    parser.add_argument('--bilstm_embed_dim', default=300, type=int)
    parser.add_argument('--bilstm_hidden_dim', default=128, type=int)
    parser.add_argument('--bilstm_num_layers', default=2, type=int)
    parser.add_argument('--bilstm_dropout', default=0.1, type=float)
    parser.add_argument('--gsbi_d_geo', default=64, type=int, help='GSBI geometry embedding dimension')
    parser.add_argument('--save_json', type=str, default=None,
                        help='Save per-sample predictions to JSON file')
    return parser


if __name__ == '__main__':
    parser = get_bilstm_gsbi_parser()
    args = parser.parse_args()
    main(args)
