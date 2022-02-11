
# @Author: Enea Duka
# @Date: 8/29/21
import torch


def prep_for_crf(x, pure_clip_lengths):
    x, x_lens, mol = restack_clips(x, pure_clip_lengths)
    return x

def prep_for_local(x, pure_nr_frames):
    x = x.detach().cpu()
    xs = []
    if len(pure_nr_frames.shape) == 2:
        pure_nr_frames = torch.t(pure_nr_frames)[0]
    for idx, pnf in enumerate(pure_nr_frames):
        vl = int(pnf.item())
        xi = x[idx].squeeze()
        xi = xi[:vl]
        xs.append(xi)

    x = torch.cat(xs, dim=0)
    x = x.cuda()
    pnfl = [16] * x.shape[0]
    pnf = torch.LongTensor(pnfl).to(x.device)
    position_ids = torch.tensor((list(range(0, x.shape[1])))) \
        .expand(1, x.shape[1]) \
        .repeat(x.shape[0], 1) \
        .to(x.device)
    return x, position_ids, pnf

def prep_for_global(x, position_ids, pure_nr_frames):
    x = x.detach().cpu()
    clip_encodings, clip_lens, mcl = restack_clips(x, pure_nr_frames)
    x.cuda()
    pure_clip_lengths = torch.LongTensor(clip_lens).to(x.device)
    num_clips = clip_encodings.shape[0]
    global_position_ids = torch.tensor(list(range(0, mcl))) \
        .expand(1, mcl) \
        .repeat(clip_encodings.shape[0], 1) \
        .to(position_ids.device)

    return clip_encodings, global_position_ids, pure_clip_lengths, num_clips

def _zeropad(tensor, size):
    n = size - tensor.shape[0] % size
    z = torch.zeros((n, tensor.shape[1])).to(tensor.device)
    return torch.cat((tensor, z), dim=0)

def restack_clips(flat_clips, pure_nr_frames):
    clips = []
    for idx, pnf in enumerate(pure_nr_frames):
        if idx == 0:
            clips.append(flat_clips[:int(pnf.item())])
        else:
            clips.append(flat_clips[int(pure_nr_frames[:idx].sum().item()):
                                    int(pure_nr_frames[:idx].sum().item()) + int(pnf.item())])

    mcl = max([c.shape[0] for c in clips])
    clip_lens = []
    for idx, clip in enumerate(clips):
        clip_lens.append(clip.shape[0])
        if clip.shape[0] < mcl:
            clips[idx] = _zeropad(clip, mcl)
    return torch.stack(clips), clip_lens, mcl