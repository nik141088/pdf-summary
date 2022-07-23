import sys


# Usage examples
# CPU: python pdf_summary.py christmas_carol.pdf 1 5 1
# GPU: python pdf_summary.py christmas_carol.pdf 1 END 0

n = len(sys.argv);
print('Pass ONLY 5 arguments: py_script_name[STR] pdf_file_name[STR], start_page[INT], end_page[INT], use_CPU[INT].'
      'A string END or MAX may be specified for end_page. If you want to use GPU set use_CPU to 0. If you get "CUDA out of memory" error then set use_CPU to 1.')
if n != 5:
    assert(False);


from transformers import pipeline
import torch
import fitz  # this is pymupdf

# -1 indicates CPU; 0 indicates GPU
device = -1
if int(sys.argv[4]) == 0 and torch.cuda.is_available():
    device = 0;



pdf_file = sys.argv[1]

with fitz.open(pdf_file) as doc:
    pdf_text = ""
    if sys.argv[3] in ["END", "MAX"]:
        sys.argv[3] = str(len(doc));
    for i in range(len(doc)):
        if i < (int(sys.argv[2]) - 1) or i > (int(sys.argv[3]) - 1):
            continue
        page = doc[i]
        # print(page)
        pdf_text += page.get_text()



ARTICLE = pdf_text


max_chunk = 128
ARTICLE = ARTICLE.replace('.', '.<eos>')
ARTICLE = ARTICLE.replace('?', '?<eos>')
ARTICLE = ARTICLE.replace('!', '!<eos>')
sentences = ARTICLE.split('<eos>')
current_chunk = 0
chunks = []
for sentence in sentences:
    # discard extremely small sentences!
    if len(sentence) <= 25:
        continue
    if len(chunks) == current_chunk + 1:
        if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
            chunks[current_chunk].extend(sentence.split(' '))
        else:
            current_chunk += 1
            chunks.append(sentence.split(' '))
    else:
        # print(current_chunk)
        chunks.append(sentence.split(' '))

for chunk_id in range(len(chunks)):
    chunks[chunk_id] = ' '.join(chunks[chunk_id])


summarizer = pipeline("summarization", model = 'D:\\nlp\\transformers\\models\\sshleifer--distilbart-cnn-12-6', device = device)
# summarizer = pipeline("summarization", model = "D:\\nlp\\transformers\\models\\google--pegasus-xsum", device = device)

res = list();
for c in chunks:
    # s = summarizer(c, min_length = 50, max_length = 100, do_sample=False)[0]['summary_text']
    s = summarizer(c, do_sample=False)[0]['summary_text']
    res.append(s)
    del s
    torch.cuda.empty_cache();

del summarizer
torch.cuda.empty_cache()

# printing output
print('\n', '-'*50, 'SUMMARY START', '-'*50, '\n')

print('\n'.join(res))

print('\n', '-'*50, 'SUMMARY END', '-'*50, '\n')



