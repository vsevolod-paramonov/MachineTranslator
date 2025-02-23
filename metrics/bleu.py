import subprocess

def bleu_score(path1, path2):

    command = f"cat {path1} | sacrebleu {path2} --tokenize none --width 2 -b"

    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return float(result.stdout.strip())