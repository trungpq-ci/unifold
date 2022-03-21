# import collections

# with open('out/MGYP000000312322/msas/uniref90_hits.sto') as f:
#     sto_str = f.read()
# with open("name.txt", "w") as f:
#     name_to_sequence = collections.OrderedDict()
#     for line in sto_str.splitlines():
#         line = line.strip()
#         if not line or line.startswith(('#', '//')):
#           continue
#         name, seq = line.split()
#         if name not in name_to_sequence:
#           name_to_sequence[name] = ''
#           f.write(name+"\n")
#     f.close()

data = "debug_database.fasta"
uniref90_name = "uniref90.name"
# data = "../openfold-main/data/uniref90/uniref90.fasta"
# name = "hits/MGYP000000312322/uniref90_hits.name"
name = "debug_hits.name"
file2 = "debug_subset.fasta"

t1 = "debug_data1.fasta"
t2 = "debug_data2.fasta"

with open(name, "r") as f, open(data, "r") as g, open(file2, "w") as h:
    for line in g:
        if not line or not line.startswith(('>')):
            continue
        data_name = line.split()[0][1:]
        for name in f:
            if name == data_name:
                h.write(line +"\n")
                while True:
                    line = g.readline()
                    if line and not line.startswith(('>')):
                        h.write(line+"\n")
                    else:
                        break
                break
        h.flush()
        f.seek(0) # go back to begining
    h.close()

#     for line in f:
#         if not line or not line.startswith(('>')):
#             continue
#         name = line.split()[0][1:]
#         h.write(name+"\n")
#         h.flush()
#     h.close()
