# # @Author: Enea Duka
# # @Date: 10/19/21
#
#
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from utils.logging_setup import logger
from tqdm import tqdm
#
def test(dataloader, model):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)

        for i, input in enumerate(tqdm(dataloader)):
            input = input[0]
            # input = input.permute(1, 0, 2)
            position_ids = torch.tensor(list(range(0, input.shape[1]))) \
                .expand(1, input.shape[1]) \
                .repeat(input.shape[0], 1)

            pure_nr_frames = torch.tensor([input.shape[1]]*input.shape[0])

            out, out_bc = model(input, position_ids, None, pure_nr_frames, high_order_temporal_features=False)
            # logits = out_bc.squeeze()
            logits = torch.mean(out_bc, 0)
            sig = logits
            pred = torch.cat((pred, sig))


        gt = np.load('list/gt-ucf.npy')
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        # precision, recall, th = precision_recall_curve(list(gt), pred)
        # pr_auc = auc(recall, precision)
        # np.save('precision.npy', precision)
        # np.save('recall.npy', recall)
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        return rec_auc.item()

if __name__ == '__main__':
    gt = np.load('list/gt-ucf.npy')
    a=1
    # pred = list(pred.cpu().detach().numpy())
    # pred = np.repeat(np.array(pred), 16)




# import matplotlib.pyplot as plt
# import torch
# from sklearn.metrics import auc, roc_curve, precision_recall_curve
# import numpy as np
# from utils.util_functions import logger
#
# def test(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         pred = torch.zeros(0).cpu()
#         print('Testing...')
#         for i, input in enumerate(dataloader):
#             input = input
#             # input = input.permute(0, 2, 1, 3)
#             # print("Path: %s ------ Len: %d" % (filename, input.shape[2]))
#             # if input.shape[1] > 3000:
#             #     input = input[:, :8]
#             # score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
#             # scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
#             logits = model(input)
#             logits = torch.squeeze(logits, 1)
#             logits = torch.mean(logits, 0)
#             # sig = torch.t(torch.softmax(torch.t(logits), dim=1))
#             sig = logits
#             # if rem_clip_len == 0:
#             #     sig = torch.repeat_interleave(sig, 16).unsqueeze(1)
#             # else:
#             #     try:
#             #         if len(sig.shape) == 0:
#             #             sig = torch.repeat_interleave(sig, rem_clip_len).unsqueeze(1)
#             #         else:
#             #             sig = torch.cat((torch.repeat_interleave(sig[:-1], 16), torch.repeat_interleave(sig[-1], rem_clip_len)), dim=0).unsqueeze(1)
#             #     except Exception:
#             #         a=1
#             pred = torch.cat((pred, sig.cpu().detach()))
#
#
#         gt = np.load('list/gt-ucf.npy')
#         pred = list(pred.detach().numpy())
#         pred = np.repeat(np.array(pred), 16)
#
#         fpr, tpr, threshold = roc_curve(list(gt), pred)
#         np.save('fpr.npy', fpr)
#         np.save('tpr.npy', tpr)
#         rec_auc = auc(fpr, tpr)
#         print('auc : ' + str(rec_auc))
#
#         precision, recall, th = precision_recall_curve(list(gt), pred)
#         pr_auc = auc(recall, precision)
#         np.save('precision.npy', precision)
#         np.save('recall.npy', recall)
#         print('Ended testing ... ')
#         torch.cuda.empty_cache()
#         return rec_auc