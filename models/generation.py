import torch.nn.functional as F
import torch


class BeamSearch:
    def __init__(self, model, max_length, width, device):
        self.model = model
        self.max_length = max_length
        self.width = width
        self.device = device

        self.bos_id = self.model.encoder.dataset.bos_id
        self.eos_id = self.model.encoder.dataset.eos_id

    @torch.inference_mode()
    def generate(self, src):
        indices = torch.tensor(self.model.encoder.dataset.text2ids(src), device=self.device).unsqueeze(0)
        encoder_outputs, hidden = self.model.encoder(indices)

        beams = [(0.0, [self.bos_id], hidden)]  

        for _ in range(self.max_length):

            candidates = []

            for score, sequence, hidden in beams:
                decoder_input = torch.tensor([[sequence[-1]]], device=self.device)

                decoder_output, next_hidden, _ = self.model.decoder.forward_step(
                    decoder_input, hidden, encoder_outputs
                )

                log_probs = F.log_softmax(decoder_output, dim=-1).squeeze(0).squeeze(0)

                top_k_probs, top_k_ids = log_probs.topk(self.width, dim=0)

                for prob, token_id in zip(top_k_probs, top_k_ids):
                    new_score = score + prob.item()  
                    new_sequence = sequence + [token_id.item()]
                    new_hidden = next_hidden

                    if token_id.item() == self.eos_id:
                        candidates.append((new_score, new_sequence, None)) 
                    else:
                        candidates.append((new_score, new_sequence, new_hidden))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:self.width]

            unfinished = [(s, seq, h) for s, seq, h in beams if seq[-1] != self.eos_id]
            if not unfinished:  
                break

        best_score, best_sequence, _ = unfinished[0]

        return self.model.decoder.dataset.ids2text(best_sequence)

class GreedyDecoding:
    def __init__(self):
        pass


