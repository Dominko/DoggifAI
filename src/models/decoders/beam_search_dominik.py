import torch
import torch.utils.data as tud
from tqdm.auto import tqdm
import warnings
from torch.nn import functional as F
import random

def beam_search_dominik(
    model, 
    X, 
    Y_init = None,
    predictions = 20,
    beam_width = 5,
    progress_bar = 0,
    input_length = 1,
    temperature=1.0,
    fixed_residues=None,
    tokenizer=None,
    device = 'cuda',
):
    with torch.no_grad():
            # print(X)
            # print(Y_init)
            num_sequences = X.shape[0]
            total_beams = beam_width * num_sequences
            # padding is all ones
            # samples = torch.ones(predictions, input_length, dtype=torch.int32).to(device)
            # probabilities = torch.zeros(predictions, 1, dtype=torch.float).to(device)

            inputs = X.repeat(beam_width, 1)
            input_sequences = Y_init.repeat(beam_width, 1).to(device)
            probabilities = torch.zeros(beam_width, 1, dtype=torch.float).to(device)
            # print(X)
            
            # print(inputs)
            # print(input_sequences)

            for l in range(input_length):
                # log = False
                # print(random.getstate())
                # print(input_sequences)
                out = model.forward(inputs, input_sequences)
                out = out[:, -1, :] / temperature
                out = F.softmax(out, dim=-1)
                # out = F.log_softmax(out, dim=-1)

                top = torch.topk(out, beam_width)
                # print(out)
                # if log:
                #     print(top.indices)
                #     print(top.values)
                
                # If we have fixed residues, we need to make sure that the we use the fixed residues
                if fixed_residues is not None:
                    if l in fixed_residues:
                        for j in range(beam_width):
                                top.indices[j, :] = tokenizer.encode(fixed_residues[l], add_special_tokens=False)[1]
                                top.values[j, :] = out[j, top.indices[j, 0]]
                                # log = True
                # if log:
                #     print(top.indices)
                #     print(top.values)
                # print(top.values.log())
                                # raise ValueError("Stop here")
                candidates = []
                candidate_probs = []

                for i in range(beam_width):
                    for j in range(len(top.indices[i])):
                        # check if the last token is an eos token
                        if input_sequences[i, -1] == tokenizer.eos_token_id or input_sequences[i, -1] == tokenizer.pad_token_id:
                            candidate = input_sequences[i].tolist() + [tokenizer.pad_token_id]
                            candidate_prob = probabilities[i]
                        else:                            
                            candidate = input_sequences[i].tolist() + [top.indices[i, j].item()]
                            candidate_prob = probabilities[i] + top.values[i, j].log()
                        candidates.append(candidate)
                        candidate_probs.append(candidate_prob)

                # if log:
                #     print(candidates)
                #     print(candidate_probs)

                # Remove duplicates
                unique_candidates = []
                unique_probs = []
                for k in range(len(candidates)):
                    if candidates[k] not in unique_candidates:
                        unique_candidates.append(candidates[k])
                        unique_probs.append(candidate_probs[k])

                # if log:
                #     print(unique_candidates)
                #     print(unique_probs)
                # print(unique_candidates)
                # print(unique_probs)

                # Sort the candidates by probability
                sorted_indices = sorted(range(len(unique_probs)), key=lambda k: unique_probs[k], reverse=True)
                sorted_candidates = [unique_candidates[idx] for idx in sorted_indices]
                sorted_probs = [unique_probs[idx] for idx in sorted_indices]

                # if log:
                #     print(sorted_candidates)
                #     print(sorted_probs)
                # print(sorted_candidates)
                # print(sorted_probs)

                # print(torch.IntTensor(sorted_candidates))
                # print(torch.FloatTensor(sorted_probs))

                # Edgecase if we have fewer sequences than the beam_width pad with same sequence
                if len(sorted_candidates) < beam_width:
                    print("Padding!")
                    sorted_candidates += [sorted_candidates[0]] * (beam_width - len(sorted_candidates))
                    sorted_probs += [sorted_probs[0]] * (beam_width - len(sorted_probs))

                # Select the top beam_width candidates
                input_sequences = torch.IntTensor(sorted_candidates[:beam_width]).to(device)
                probabilities = torch.FloatTensor(sorted_probs[:beam_width]).to(device)
                # print(inputs)
                # print(probabilities)
                

                # if log:
                #     print(input_sequences)
                #     print(probabilities)

                # if l == 100:
                #     raise ValueError("Stop here")
            
            return input_sequences, probabilities