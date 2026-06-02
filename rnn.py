import torch
import numpy as np
import urllib.request
from pathlib import Path


print('hello')
# hyperparams
hidden_size = 512
embed_size = 64
seq_length = 128
learning_rate = 1e-2
min_learning_rate = 1e-4
time_to_decay = 10000
time_to_sample = 1000
learning_rate_decay = 0.98
how_much_to_sample = 6000
max_iteration = 11000000
temp = 0.8
checkpoint_every = 300000

# hs 64 (not very relevant, since 70k):
# iter 70000, loss_sum: 260.480, loss_ce: 2.0350, batch_ce: 2.2335, ppl: 7.65, grad_l2: 115.43, clip_frac: 0.0021

# hs 128 embed 128:
# iter 10000, loss_sum: 284.910, loss_ce: 2.2259, batch_ce: 2.2756, ppl: 9.26, grad_l2: 95.02, clip_frac: 0.0004

# hs 128 seq embed 64
# iter 10000, loss_sum: 271.246, loss_ce: 2.1191, batch_ce: 2.1240, ppl: 8.32, grad_l2: 124.65, clip_frac: 0.0001

# hs 256 seq embed 64 seq_length 128
# iter 10000, loss_sum: 250.745, loss_ce: 1.9589, batch_ce: 1.9153, ppl: 7.09, grad_l2: 158.95, clip_frac: 0.0000

# hs 256 seq embed 64 seq_length 64
# iter 10000, loss_sum: 133.921, loss_ce: 2.0925, batch_ce: 2.0066, ppl: 8.11, grad_l2: 106.16, clip_frac: 0.0001

# hs 256 seq embed 64 seq_length 256
# iter 10000, loss_sum: 486.635, loss_ce: 1.9009, batch_ce: 1.9498, ppl: 6.69, grad_l2: 222.14, clip_frac: 0.0018

# hs 256 embed 64 seq_length 128
# iter 500000, loss_sum: 189.075, loss_ce: 1.4771, batch_ce: 1.6370, ppl: 4.38, grad_l2: 319.49, clip_frac: 0.0066

# hs 512 embed 64 seq_length 128
# iter 953000, loss_sum: 178.392, loss_ce: 1.3937, batch_ce: 1.3190, ppl: 4.03, grad_l2: 290.93, clip_frac: 0.0000


# load data
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent / "smol_rnn_data"
LOCAL_PATH = DATA_DIR / "tinyshakespeare.txt"
DATASET_NAME = LOCAL_PATH.stem
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
if not LOCAL_PATH.is_file():
    print('download data...')
    with urllib.request.urlopen(DATA_URL) as response:
        LOCAL_PATH.write_bytes(response.read())
else:
    print('data was found on a disk, no download needed')

text = LOCAL_PATH.read_text(encoding="utf-8")
# text = LOCAL_PATH.read_text(encoding="utf-8")[:how_much_to_sample]
if not text:
    raise RuntimeError("failed to get any text data")
print(f"we got {len(text.split()):,} words, it's about {len(text):,} chars")

# build the vocab
chars = sorted(set[str](text))
vocab_size = len(chars)
print(f"{vocab_size} unique chars in text aka vocab size")
char_to_idx = {chars[i]: i for i in range(vocab_size)}
ix_to_char = {i: chars[i] for i in range(vocab_size)}
print(char_to_idx)
print(ix_to_char)

# encoding data
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.int64)
assert data.numel() == len(text)


def init_param(dim1, dim2, reg=.01): return np.random.randn(dim1, dim2)*reg
def init_bias(dim1): return np.zeros((dim1, 1))


# model params
E = init_param(embed_size, vocab_size)
Wxh = init_param(hidden_size, embed_size)
Whh = init_param(hidden_size, hidden_size)
bh = init_bias(hidden_size)

Why = init_param(vocab_size, hidden_size)
by = init_bias(vocab_size)


def fps(x_t, Wxh, Whh, hs_prev_t, bh):
    # # hiddens
    # hs[i] = np.tanh(np.dot(Wxh, xs[i]) + np.dot(Whh, hs[i-1]) + bh)
    # # unnormalized log probabilities for next chars
    # ys[i] = np.dot(Why, hs[i]) + by
    # # actual probabilities for next chars
    # ps[i] = np.exp(ys[i]) / np.sum(np.exp(ys[i]))
    # -------------------------------------------------

    h = np.tanh(np.dot(Wxh, x_t) + np.dot(Whh, hs_prev_t) + bh)

    return h


def loss_fn(inputs, targets, hprev):
    xs, ys, ps = {}, {}, {}
    input_ix = {}
    h_t = {}

    h_t[-1] = np.copy(hprev)

    loss = 0
    inputs_len = len(inputs)

    # forward pass
    for i in range(inputs_len):
        token_ix = int(inputs[i])
        input_ix[i] = token_ix
        xs[i] = E[:, token_ix: token_ix + 1]

        h_t[i] = fps(xs[i], Wxh, Whh, h_t[i-1], bh)

        ys[i] = np.dot(Why, h_t[i]) + by

        exp_scores = np.exp(ys[i] - np.max(ys[i]))
        ps[i] = exp_scores / np.sum(exp_scores)
        loss += -np.log(ps[i][int(targets[i]), 0])

    # backward pass: just manual compute of grads
    dWxh, dWhh = np.zeros_like(Wxh), np.zeros_like(Whh)
    dE = np.zeros_like(E)
    dWhy = np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(h_t[0])

    for j in reversed(range(inputs_len)):
        dy = np.copy(ps[j])
        # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dy[int(targets[j])] -= 1
        dWhy += np.dot(dy, h_t[j].T)
        dby += dy

        dh = np.dot(Why.T, dy) + dh_next
        dhraw = (1 - h_t[j] * h_t[j]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[j].T)
        dWhh += np.dot(dhraw, h_t[j - 1].T)
        dh_next = np.dot(Whh.T, dhraw)
        dx = np.dot(Wxh.T, dhraw)
        dE[:, input_ix[j]] += dx[:, 0]

    for dparam in [dE, dWxh, dWhh, dbh, dWhy, dby]:
        # clip to mitigate exploding gradients
        np.clip(dparam, -5, 5, out=dparam)

    return (
        loss,
        dE,
        dWxh,
        dWhh,
        dbh,
        dWhy,
        dby,
        h_t[inputs_len - 1],
    )


def sample(h, seed_ix, sample_len, temperature=temp):
    seed_ix_int = int(seed_ix)
    x = E[:, seed_ix_int: seed_ix_int + 1]
    ixes = [seed_ix_int]

    for _ in range(sample_len):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        y = y / temperature
        exp_scores = np.exp(y - np.max(y))
        p = exp_scores / np.sum(exp_scores)
        predicted_ix = np.random.choice(range(vocab_size), p=p.ravel())
        ixes.append(predicted_ix)

        # next iteration symbol
        x = E[:, predicted_ix: predicted_ix + 1]

    return ixes


iteration, data_pointer = 0, 0
# memory variables for Adagrad
mE = np.zeros_like(E)
mWxh, mWhh, mbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
mWhy, mby = np.zeros_like(Why), np.zeros_like(by)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length
hprev = None


def save_checkpoint(iteration, data_pointer, hprev):
    checkpoint_path = CHECKPOINT_DIR / \
        f"smol_rnn_{DATASET_NAME}_iter_{iteration:07d}.npz"
    np.savez_compressed(
        checkpoint_path,
        E=E,
        Wxh=Wxh,
        Whh=Whh,
        bh=bh,
        Why=Why,
        by=by,
        mE=mE,
        mWxh=mWxh,
        mWhh=mWhh,
        mbh=mbh,
        mWhy=mWhy,
        mby=mby,
        hprev=hprev,
        chars=np.array(chars),
        iteration=iteration,
        data_pointer=data_pointer,
        dataset_name=DATASET_NAME,
        data_path=str(LOCAL_PATH),
        smooth_loss=smooth_loss,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        embed_size=embed_size,
        seq_length=seq_length,
        temp=temp,
    )
    print(f"saved checkpoint: {checkpoint_path}")


while iteration < max_iteration:
    is_time_to_decay = iteration > 0 and iteration % time_to_decay == 0
    if is_time_to_decay:
        learning_rate = max(min_learning_rate,
                            learning_rate * learning_rate_decay)

    is_sample_iteration = iteration % time_to_sample == 0
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if data_pointer + seq_length + 1 >= len(data) or iteration == 0:
        # reset RNN memory
        hprev = np.zeros((hidden_size, 1))
        data_pointer = 0
    inputs = data[data_pointer: data_pointer + seq_length]
    targets = data[data_pointer + 1: data_pointer + seq_length + 1]

    # sample from the model time to time
    if is_sample_iteration:
        sample_xs = sample(hprev, inputs[0], 200)
        start = str(ix_to_char[int(inputs[0].item())])

        txt = ''.join(ix_to_char[ix]
                      for ix in sample_xs)
        print('----\n%s\n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    (
        loss,
        dE,
        dWxh,
        dWhh,
        dbh,
        dWhy,
        dby,
        hprev,
    ) = loss_fn(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if is_sample_iteration:
        batch_ce = loss / seq_length
        smooth_ce = smooth_loss / seq_length
        ppl = np.exp(np.minimum(smooth_ce, 20.0))

        grad_sq_sum = 0.0
        clipped_count = 0
        total_grad_count = 0
        for grad in [dE, dWxh, dWhh, dbh, dWhy, dby]:
            grad_sq_sum += float(np.sum(grad * grad))
            clipped_count += int(np.sum(np.abs(grad) >= 4.999))
            total_grad_count += grad.size
        grad_l2 = np.sqrt(grad_sq_sum)
        clip_frac = clipped_count / total_grad_count

        print(
            "iter %d, lr: %.6f, loss_sum: %.3f, loss_ce: %.4f, batch_ce: %.4f, ppl: %.2f, grad_l2: %.2f, clip_frac: %.4f"
            % (iteration, learning_rate, smooth_loss, smooth_ce, batch_ce, ppl, grad_l2, clip_frac)
        )

    for param, dparam, mem in zip(
        [E, Wxh, Whh, bh, Why, by],
        [dE, dWxh, dWhh, dbh, dWhy, dby],
        [mE, mWxh, mWhh, mbh, mWhy, mby],
    ):

        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    data_pointer += seq_length
    iteration += 1

    if iteration % checkpoint_every == 0:
        save_checkpoint(iteration, data_pointer, hprev)
