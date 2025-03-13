import torch.nn.functional as F
import torch


class BeamSearch:
    def __init__(self, model, max_length, width, length_penalty, device):

        self.model = model
        
        self.max_length = max_length
        self.width = width
        self.device = device

        self.length_penalty = length_penalty

        self.bos_id = self.model.de.bos_id
        self.eos_id = self.model.de.eos_id

    @torch.inference_mode()
    def generate(self, src):
        indices = torch.tensor(self.model.de.text2ids(src), device=self.device).unsqueeze(0)
        encoder_output, encoder_mask = self.model.encode(indices)

        beams = [(0.0, [self.bos_id])]  
        finished_beams = []

        for _ in range(self.max_length):

            candidates = []

            for score, sequence in beams:
                decoder_input = torch.tensor(sequence, device=self.device).unsqueeze(0)

                decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input)

                decoder_output = self.model.out(decoder_output[:, -1, :])

                log_probs = F.log_softmax(decoder_output, dim=-1)

                top_k_probs, top_k_ids = log_probs.topk(self.width, dim=-1)

                for prob, token_id in zip(top_k_probs.squeeze(0), top_k_ids.squeeze(0)):
                    new_score = score + prob.item()  
                    new_sequence = sequence + [token_id.item()]

                    new_score = new_score / (len(new_sequence) ** self.length_penalty)

                    if token_id.item() == self.eos_id:
                        finished_beams.append((new_score, new_sequence)) 
                    else:
                        candidates.append((new_score, new_sequence))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:self.width]

            if not beams:  
                break

        if finished_beams:
            _, best_seq = max(finished_beams, key=lambda x: x[0])
        else:
            _, best_seq = max(beams, key=lambda x: x[0])

        return self.model.en.ids2text(best_seq)

class GreedyDecoding:
    def __init__(self, model, max_length, device):
        self.model = model
        self.max_length = max_length

        self.device = device

    @torch.inference_mode()
    def generate(self, src):
        
        indices = torch.tensor(self.model.de.text2ids(src), device=self.device).unsqueeze(0)
        
        encoder_output, encoder_mask = self.model.encode(indices)

        decoder_input = torch.tensor([[self.model.de.bos_id]], device=self.device, dtype=torch.long)

        generated_tokens = []

        for _ in range(self.max_length):
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input)

            pred = self.model.out(decoder_output[:, -1, :])

            next_token_id = self.model.out(decoder_output[:, -1, :]).argmax(dim=-1).item()
            generated_tokens.append(next_token_id)

            next_token = torch.tensor([[next_token_id]], device=self.device)
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if next_token_id == self.model.de.eos_id:
                break

        return self.model.en.ids2text(generated_tokens)
