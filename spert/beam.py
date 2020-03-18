import torch


class BeamSearch(object):

    def __init__(self, beam_size=3):

        self._beam_size = beam_size

        self._prev_ks = [] # visited states
        self._curr_ys = [torch.LongTensor(beam_size).fill_(0)]

        self._curr_scores = torch.FloatTensor(beam_size).zero_()
        self._all_scores = []


    def advance(self, next_logits):
        # next_logits: (beam_size, vocab_size)

        vocab_size = next_logits.shape[-1]
        # print("vocab size:", vocab_size)
        # print("logits:", next_logits)
        if len(self._prev_ks) > 0:
            beam_scores = next_logits + self._curr_scores.unsqueeze(1).expand_as(next_logits)
        else:
            beam_scores = next_logits[0]

        flat_beam_scores = beam_scores.view(-1)
        # print("flat beam scores:", flat_beam_scores)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self._beam_size, dim=0, largest=True, sorted=True)
        self._curr_scores = top_scores
        self._all_scores.append(self._curr_scores)
        # print("top score ids:", top_score_ids)
        prev_k = top_score_ids / vocab_size # beam index
        # print("prev k:", prev_k)
        curr_y = top_score_ids - prev_k * vocab_size 
        # print("curr_y:", curr_y)
        
        self._prev_ks.append(prev_k)
        self._curr_ys.append(curr_y)

        return

    @property
    def get_beam_size(self):
        return self._beam_size
    
    @property
    def get_curr_state(self):
        return self._curr_ys[-1]

    @property
    def get_curr_origin(self):
        return self._prev_ks[-1]

    @property
    def get_curr_scores(self):
        return self._curr_scores

    @property
    def get_best_path(self):

        length = len(self._all_scores)
        # print("prev pointers:", self._prev_ks)
        bps = [self._prev_ks[i][0] for i in range(length)]
        preds = torch.tensor([self._curr_ys[i+1][bps[i]].item() for i in range(length)])
        # print("preds:", preds)
        scores = torch.tensor([self._all_scores[i][bps[i]].item() for i in range(length)])
        # print("scores:", [self._all_scores[i] for i in range(length)])
        return scores, preds
    
    
    


    

