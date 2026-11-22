"""Exact raw100-bit verifier, avoiding float32 90/100 < Python0.9."""
def correct_bits(record,nbits=100):
    score=float(record['bit_accuracy']);count=round(score*nbits)
    if not 0<=count<=nbits or abs(score-count/nbits)>1e-6:raise ValueError('Expected integer-grid raw bit accuracy')
    return count

def verified(record):
    return correct_bits(record,100)>=90
