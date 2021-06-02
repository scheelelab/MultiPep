#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import pickle
import re
import numpy as np
np.random.seed(123) #42
import random
random.seed(123)# 42
import seaborn as sns
import matplotlib.pyplot as plt

def items_per_key(dct):
    print("\n")
    for k,v in dct.items():
        print(k,len(v))

def overlapping_items(dct):
    for k,v in dct.items():
        print("\n",k,len(v))
        for kk,vv in dct.items():
            if k != kk:
                inter = len(set(v).intersection(set(vv)))
                if inter > 0:
                    print(k, kk,inter)


def find_length_dist(nm, sq):
    lens = {}
    for i in sq:
        ln = len(i)
        if ln in lens:
            lens[ln] += 1
        else:
            lens[ln] = 1

    n1 = []
    n2 = []
    for k,v in lens.items():
        n1.append(k)
        n2.append(v)
    
    c = list(zip(n1,n2))
    c.sort()
    a1,b1 = zip(*c)

    plt.figure(figsize=(10,7))
    plt.bar(a1,[int(i) for i in b1])
    plt.title("Peptide length distribution of class '{}'".format(nm), fontsize=16)
    plt.ylabel("Counts", fontsize=14)
    plt.xlabel("Peptide length", fontsize=14)
    plt.xlim(0,200)
    #plt.show()
    plt.savefig("Peptide length {}".format(nm)+".png")
    plt.close()


def aa_counter(nm, sq1, vocab):
    h = {}
    for i in sq1:
        for ii in i:
            if ii in h:
                h[ii] += 1
            if ii not in h:
                h[ii] = 1

    n1 = []
    n2 = []
    vocab.sort()
    for i in vocab:
        if i in h:    
            n1.append(i)
            n2.append(h[i])
        else:
            n1.append(i)
            n2.append(0)

    c = list(zip(n1,n2))
    c.sort()
    a1,b1 = zip(*c)
    vocab.sort()
    plt.figure(figsize=(10,7))
    plt.bar(n1,b1)
    plt.title("Amino acids distribution of class '{}'".format(nm), fontsize=16)
    plt.ylabel("Counts", fontsize=14)
    plt.xlabel("Amino acids", fontsize=14)
    #plt.show()
    plt.savefig("Amino_acids {}".format(nm)+".png")
    plt.close()


def find_overlapping_items(dct):
    new_dct = {}
    for k,v in dct.items():
        tmp_dct = {}
        for kk,vv in dct.items():
            if k != kk:
                inter = len(set(v).intersection(set(vv)))
                if inter > 0:
                    tmp_dct[kk] = inter
        new_dct[k] = tmp_dct
    return new_dct


def find_overlapping_items_lst(dct):
    new_dct = {}
    for k,v in dct.items():
        tmp_dct = []
        for kk,vv in dct.items():
            #if k != kk:
            inter = len(set(v).intersection(set(vv)))
            if inter > 0:
                tmp_dct.append([inter, kk])
        tmp_dct.sort()
        new_dct[k] = tmp_dct
    return new_dct


removed_sequences = []

vocab = ['A',
 'R',
 'N',
 'D',
 'C',
 'Q',
 'E',
 'G',
 'H',
 'I',
 'L',
 'K',
 'M',
 'F',
 'P',
 'S',
 'T',
 'W',
 'Y',
 'V',]
# 'B',
# 'Z',
# 'J',
# 'U',
# 'O',
 #'X']

########################################################
########################################################
########################################################
# CAMP data


# reading all downloaded CAMP data
al = pd.read_csv("camp_data/all_camp.csv", sep="\t", index_col=0)


dec = {"antioncogenic":"anticancer",
        "antiparasitic":"antiparasite",
        "antitumor":"anticancer",
        'antitumour':"anticancer",
        "antiviral":"antivirus"}



camp_sq_data = {}
sq = list(al["Seqence"])
acti = list(al["Activity"])
val = list(al["Validation"])

for i in range(len(sq)):
    if val[i] == "Experimentally Validated": # Only using Exp. validated
        tmp = acti[i].replace(".",",").strip().split(",")
        tmp = [ii.lower().strip() for ii in tmp]
        tmp1 = []
        for ii in tmp:
            if ii in dec:
                tmp1.append(dec[ii])
            else:
                tmp1.append(ii)
        assert len(tmp1) == len(tmp)
        tmp = tmp1
        sqq = sq[i].strip().upper()
        
        if len(set(list(sqq)).difference(vocab)) == 0: ## Check for vocab agreement
            if len(sqq) > 1 and len(sqq) <= 200: ## Size check
                if sqq in camp_sq_data:
                    camp_sq_data[sqq].update(set(tmp))
                if sqq not in camp_sq_data:
                    camp_sq_data[sqq] = set(tmp)
        else:
            removed_sequences.append(sqq)
            
with open("camp_dct_data.pkl","wb") as f:   #Saving the data
    pickle.dump(camp_sq_data,f)


######### Exp. validated


########################################################
########################################################
########################################################
## LAMP data


# reading all LAMP data
nmsl = glob.glob1("lamp_data/","*")
with open("lamp_data/"+nmsl[0],"rb") as f:
    data = f.read().decode(errors='replace')
    data = data.split("\n")


# Gathering information about the sequences
al_seqs = []
for i in data:
    if "<AMP " in i:
        tmp = {}
    if "<Activity>" in i:
        # Extracting bioactivities
        if len(i.replace("/","").split("<Activity>")) == 3:
            tmp["activity"] = i.replace("/","").split("<Activity>")[-2]
            
    if "<Source>" in i:
        tmp["animal"] = i.replace("/","").split("<Source>")[-2]
            
    if "<SourceSet>" in i:
        tmp["source"] = i.replace("/","").split("<SourceSet>")[-2]
            
    if "<Sequence>" in i:
        tmp["seq"] = i.replace("/","").split("<Sequence>")[-2]
        
    if "<Dataset>" in i:
        tmp["type"] = i.replace("/","").split("<Dataset>")[-2]        

    if "</AMP>" in i:
        al_seqs.append(tmp)
        tmp = {}
            
with open("lamp_data.pkl","wb") as f:
    pickle.dump(al_seqs,f)

## Only extracting exp validated or patented sequences
lamp_dct = {}
for i in al_seqs:
    if i["type"] == "Experimental" or "Patent":
        sqq = i["seq"].strip().upper()
        if len(set(list(sqq)).difference(vocab)) == 0:  # vocabulary check
            if len(sqq) > 1 and len(sqq) <= 200:        # size check
                if sqq in lamp_dct:
                    lamp_dct[sqq].update(set(i["activity"].split(",")))
                if sqq not in lamp_dct:
                    lamp_dct[sqq] = set(i["activity"].split(","))
        else:
            removed_sequences.append(sqq)

## Reducing class number:
c = set()
c1 = set()
co = {}
for k,v in lamp_dct.items():
    tmp = list(v)
    tmp = [i.lower().strip() for i in tmp]
    tmp.sort()
    t = ""
    for i in tmp:
        t += i+"_"
    c.add(t)
    c1.update(set(tmp))
    if t in co:
        co[t] += 1
    if t not in co:
        co[t] = 1



# the overall classes from LAMP2
over = ["antibacterial", "anticancer","antifungal" , "antivirus", 
        "antiparasite", "toxic", #, #"hemolytic",
        "antihypertensive", "antiinflammatory", "antimicrobial", "cellcellsignaling",
        'chemotactic','drugdeliveryvehicle','enzymeinhibitor',"insecticidal",
        "hemolytic", "antioxidant", "bloodbrainbarrier"] 


# Name-seqments to be associated with the classes above
genres = [
{"biofilm", "gram","bacterial", "mollicute"}, 

{"cancer", "tumor", "tumour"},

{"fung"}, 

{"virus", "hiv", "viral"},

{"parasit", "malaria","plasmod", "antileishmania", 'antiprotozoal',
 'antitrypanosomic', 'nematode'},

{"cytoto", "toxic",},

{"hypert"},

{"inflam"},

{"microb"},

{"cell-c", "cellp"},

{"chemot"},

{"drugd"},

{"enzymeinh"},

{"insect"}, {"hemolytic"}, {"oxid"}, {"bloodbrainbarrier"}]

lamp_broad_cls = {}
for k,v in lamp_dct.items():
    tmp = list(v)
    tmp = [i.lower().strip() for i in tmp if len(i) > 0]
    for i in tmp:
        for jj,ii in enumerate(genres):
            for iii in ii:
                if iii in i:
                    if over[jj] in lamp_broad_cls:
                        lamp_broad_cls[over[jj]].add(k)
                         
                    if over[jj] not in lamp_broad_cls:
                        lamp_broad_cls[over[jj]] = {k}

for k,v in lamp_broad_cls.items():
    print(k,len(v))


with open("lamp_broad_cls_data.pkl","wb") as f:
    pickle.dump(lamp_broad_cls,f)


lamp_seq_dct = {}
vocc = set()
for k,v in lamp_broad_cls.items():
    for i in v:
        if len(set(list(i.upper())).difference(vocab)) > 0:
            removed_sequences.append(i)
            #print(i)
            continue
        vocc.update(set(list(i.upper())))
        if i in lamp_seq_dct:
            lamp_seq_dct[i.upper()].add(k)
        if i not in lamp_seq_dct:
            lamp_seq_dct[i.upper()] = {k}

with open("lamp_seq_vs_class.pkl","wb") as f:
    pickle.dump(lamp_seq_dct,f)



"""
Exluded classes:
    
c2 = set()
for i in c1:
    for ii in genres:
        for iii in ii:
            if iii in i:
                c2.add(i)
c1.difference(c2)


{'',
 'a',
 'anti',
 'anti-',
 'antiye',
 'ho',
 'mammalian cell',
 'mammaliancells',
 'na.',
 'quorumsensing.',
 'surfaceimmobilized'}

"""

######### Exp. validated

#############################################################
#############################################################
#############################################################


# Extracting APD3 data
nms = glob.glob1("APD3_data/","*")
nms.remove("INFO.txt")
apd3_dct = {}
for n in nms:
    print(n)
    x = n.split(".")[0]
    apd3_dct[x] = []
    tmp = 0
    with open("APD3_data/"+n,"r") as f:
        for i in f:
            if tmp == 0:
                print(i.strip())
                tmp += 1
            else:
                line = i.split("\t")
                if len(line) > 1:
                    apd3_dct[x].append(line[-1])
    print(len(apd3_dct[x]),"\n")
    
with open("apdr_dct_data.pkl","wb") as f:
    pickle.dump(apd3_dct,f)
            

# Converting to dict and making vobaulary check
apd3_seq_dct = {}
vocc1 = set()
for k,v in apd3_dct.items():
    for i in v:
        sqq = i.strip().upper()
        if len(set(list(sqq)).difference(vocab)) > 0:
            removed_sequences.append(sqq)
            print(i)
            continue
        vocc1.update(set(list(sqq)))
        if sqq in apd3_seq_dct:
            apd3_seq_dct[sqq].add(k)
        if sqq not in apd3_seq_dct:
            apd3_seq_dct[sqq] = {k}

with open("apdr_seq_dct.pkl","wb") as f:
    pickle.dump(apd3_seq_dct,f)
    


    
#############################################################
#############################################################
#############################################################
    
    
## finding unique sequences, filter them according to size and aa-letter content
nms = glob.glob1("SATPdb/","*.fa")
satp_class_dct = {}
for i in nms:
    tmp = 0
    cl = i.split(".")[0]
    with open("SATPdb/"+i, "r",encoding='utf-8') as f:
        for ii in f:
            tmp += 1
            if ii[0] != ">":
                line = ii.strip().upper()
                if len(set(list(line)).difference(vocab)) == 0:
                    if len(line) > 1 and len(line) <= 200:
                        if cl in satp_class_dct:
                            satp_class_dct[cl].add(line)
                        if cl not in satp_class_dct:
                            satp_class_dct[cl] = {line}
                    else:
                        removed_sequences.append(line)
                else:
                    removed_sequences.append(line)
    print(i, tmp, tmp/2)
    

nms = glob.glob1("SATPdb/","*.txt")
for i in nms:
    cl = i.split(".")[0]
    with open("SATPdb/"+i, "r") as f:
        for ii in f:
            line = ii.strip().split("\t")
            if "satpdb" in line[0]:
                line = line[1].upper()
                if len(set(list(line)).difference(vocab)) == 0:
                    if len(line) > 1 and len(line) <= 200:
                        if cl in satp_class_dct:
                            satp_class_dct[cl].add(line)
                        if cl not in satp_class_dct:
                            satp_class_dct[cl] = {line}
                    else:
                        removed_sequences.append(line)
                else:
                    removed_sequences.append(line)


## printing size of classes
print("\nunique sequences per class:")
items_per_key(satp_class_dct)
print("\noverlapping items:")
overlapping_items(satp_class_dct)


#for k,v in satp_class_dct.items():
#    find_length_dist(v)

#for k,v in satp_class_dct.items():
#    aa_counter(v, vocab)



## saving the data
satp_seq_dct = {}
for k,v in satp_class_dct.items():
    for i in v:
        if i in satp_seq_dct:
            satp_seq_dct[i].add(k)
        if i not in satp_seq_dct:
            satp_seq_dct[i] = {k}

with open("satp_seq_dct.pkl","wb") as f:
    pickle.dump(satp_seq_dct,f)


######### Exp. validated 


#############################################################
#############################################################
#############################################################


nms = glob.glob1("DBAASP/","*")
nms.remove("info.txt")
nms.remove("remove")

## removing sequences if they have letters not in my vocabulary
## finding unique peptides
## finding sequences in size range between 2 and 200 aa
rest = []
dbaasp_class_dct = {}
for i in nms:
    print(i)
    tmp = pd.read_csv("DBAASP/"+i)
    sqs = list(tmp["SEQUENCE"])
    sqs = [ii.upper() for ii in sqs if type(ii) == str]
    sqs1 = set()
    for line in sqs:
        if len(set(list(line)).difference(vocab)) == 0:
            if len(line) > 1 and len(line) <= 200:
                sqs1.add(line)
        else:
            rest.append(line)
            removed_sequences.append(line)
    dbaasp_class_dct[i] = sqs1

dbaasp_class_dct1 = {}
dbaasp_class_dct1["antibacterial"] = set(list(dbaasp_class_dct["peptides_gramplus_grammin.csv"]) + list(dbaasp_class_dct["peptides_mollicutes_bacteria.csv"]))
dbaasp_class_dct1["anticancer"] = dbaasp_class_dct["peptides_cancer.csv"]
dbaasp_class_dct1["antifungal"] = dbaasp_class_dct["peptides_fungus.csv"]
dbaasp_class_dct1["antivirus"] = dbaasp_class_dct["peptides_virus.csv"]
dbaasp_class_dct1["antiparasite"] = set(list(dbaasp_class_dct["peptides_parasite.csv"]) + list(dbaasp_class_dct["peptides_parasite_nematode_protista.csv"]))
dbaasp_class_dct1["insectcides"] = dbaasp_class_dct["peptides_insecticides.csv"]

dbaasp_class_dct = dbaasp_class_dct1



items_per_key(dbaasp_class_dct)
#for k,v in dbaasp_class_dct.items():
#    find_length_dist(v)
overlapping_items(dbaasp_class_dct)
#for k,v in dbaasp_class_dct.items():
#    aa_counter(v, vocab)


## saving the data
dbaasp_seq_dct = {}
for k,v in dbaasp_class_dct.items():
    for i in v:
        if i in dbaasp_seq_dct:
            dbaasp_seq_dct[i].add(k)
        if i not in dbaasp_seq_dct:
            dbaasp_seq_dct[i] = {k}

with open("dbaasp_seq_dct.pkl","wb") as f:
    pickle.dump(dbaasp_seq_dct,f)


######### Exp. validated 



#############################################################
#############################################################
#############################################################

## BIOPEP-UWM
pep = pd.read_csv("biopepdata/biopep_data.txt",sep="\t")


pre_seq = list(pep["Sequence"])
act = list(pep["Activity "])
assert len(act) == len(pre_seq)


act_overview = list(set(act))
act_overview.sort()

act_dct = {}
for i in act_overview:
    act_dct[i] = set()

print("Number of unique sequences: ",len(set(pre_seq)))
print("Number of unique activities: ", len(set(act)))



# Iterating through the sequences and removing all non str instances
# and all strings that contain lowercase letters
# Removing all strings that occur twice under the same activity
w = ""
seq_strings = []
seq_acts = []
n500 = 0
for j,i in enumerate(pre_seq):
    if type(i) == str:
        if len(i) > 0:
            string = i.upper()
            if len(set(list(string)).difference(vocab)) == 0:
                if len(string) > 1 and len(string) <= 200:
                    act_dct[act[j]].add(string)
                    seq_strings.append(string)
                    seq_acts.append(act[j])
                else:
                    removed_sequences.append(string)
                    print(2,string)
            else:
                removed_sequences.append(string)
                print(1,string)


print(len(act_dct))
for k,v in act_dct.items():
    print(k, len(v), len(set(v)))
    assert len(v) == len(set(v))

## Use classes that have more than 100 pepides or can be merged with 
## similar classes from other databases
class_decoder = {
        'ACE inhibitor':['ACE inhibitor'],   
        'dipeptidyl peptidase inhibitor':['dipeptidyl peptidase III inhibitor', 'dipeptidyl peptidase IV inhibitor'],
        "antibacterial":["antibacterial"],
        "anticancer":["anticancer"],
        "antifungal":["antifungal"],
        "antiinflammatory":["anti inflammatory"],
        "antioxidative": ["antioxidative"],
        "antivirus":["antiviral"],
        "neuropeptide":["neuropeptide"],
        "opioid":["opioid", "opioid agonist"],
        "toxic":["celiac toxic", "toxic", "embryotoxic"],
        "haemolytic":["haemolytic"],
        "hypotensive":["hypotensive"],
        "antidiabetic":["antidiabetic"],
        "chemotactic" : ["chemotactic"]
}


biopep_broad_dct = {}
for i in list(act_dct.keys()):
    tmp = 0
    if len(act_dct[i]) >0:
        for k,v in class_decoder.items():
            if i.strip() in v:
                tmp += 1
                if k in biopep_broad_dct:
                    biopep_broad_dct[k].update(set(act_dct[i]))
                if k not in biopep_broad_dct:
                    biopep_broad_dct[k] = set(act_dct[i])
        if tmp == 0:    
            print(i, len(act_dct[i]))


items_per_key(biopep_broad_dct)
#for k,v in biopep_broad_dct.items():
#    find_length_dist(v)
overlapping_items(biopep_broad_dct)
#for k,v in biopep_broad_dct.items():
#    aa_counter(v, vocab)


## 4) convering to dict and saving
biopep_seq_dct = {}
for k,v in biopep_broad_dct.items():
    for i in v:
        if i in biopep_seq_dct:
            biopep_seq_dct[i].add(k)
        if i not in biopep_seq_dct:
            biopep_seq_dct[i] = {k}

with open("biopep_seq_dct.pkl","wb") as f:
    pickle.dump(biopep_seq_dct,f)





"""
included classes:

{'ACE inhibitor',
 'antibacterial',
 'anticancer',
 'antidiabetic',
 'antifungal',
 'antiinflammatory',
 'antioxidative',
 'antivirus',
 'chemotactic',
 'dipeptidyl peptidase inhibitor',
 'haemolytic',
 'hypotensive',
 'neuropeptide',
 'opioid',
 'toxic'}

"""

######### Exp. validated - curated



#############################################################
#############################################################
#############################################################

## Peptide_DB


print("\nExtracting data and creating sets with unique sequences")
data_nm = ['peptide_db/ant_fr.txt',
 'peptide_db/ant_mic.txt',
 'peptide_db/cyto_grow.txt',
 'peptide_db/pephor.txt',
 'peptide_db/tox_ven.txt']


def get_seqs(name):
    sqs = []
    with open(name,"r") as f:
        for i in f:
            lin = i.strip()
            if len(lin) != 0:
                sqs.append(lin)
    return sqs


petidedb_dec = {'peptide_db/ant_fr.txt': "antifreeze",
                'peptide_db/ant_mic.txt':"antimicrobial",
                'peptide_db/cyto_grow.txt':"cytokines_growthfactors",
                'peptide_db/pephor.txt':"peptidehormone",
                'peptide_db/tox_ven.txt':"toxic"
        }

## counting unique peptides
print("Unique sequences per class:")
data_dct = {}
for i in data_nm:
    tmp = set(get_seqs(i))
    data_dct[petidedb_dec[i]] = set(tmp)

items_per_key(data_dct)
print("---------------")
print("Overlaps:")
overlapping_items(data_dct)


## remove sequences if they have letters not in my vocabulary
### Making dict where all sequnces are associated with all their classes
peptidedb_seq_dct = {}
for k,v in data_dct.items():
    for i in v:
        line = i.upper()
        if len(i) > 1 and len(i) < 201:
            if len(set(list(line)).difference(vocab)) == 0:
                if i in peptidedb_seq_dct:
                    peptidedb_seq_dct[i].add(k)
                elif i not in peptidedb_seq_dct:
                    peptidedb_seq_dct[i] = {k}
            if len(set(list(line)).difference(vocab)) > 0:
                removed_sequences.append(line)
        else:
            removed_sequences.append(line)



## saving

with open("peptidedb_seq_dct.pkl","wb") as f:
    pickle.dump(peptidedb_seq_dct,f)


######### Exp. validated  --> not known!!!


#############################################################
#############################################################
#############################################################


neuro = pd.read_excel("neuropedia/Database_NeuroPedia_063011.xls", header=[0,1])

## remove sequences if they have letters not in my vocabulary
seqs = list(neuro[( 'Unnamed: 0_level_0', 'Amino acid Sequence')])
seqs = [i.upper() for i in seqs if len(set(list(i.upper())).difference(set(vocab))) == 0]
removed_sequences += [i.upper() for i in seqs if len(set(list(i.upper())).difference(set(vocab))) != 0]


## counting unique peptides
print("number of unique peptides:", len(seqs))

## removing sequences if they are not in the size range 2-200 aa
sq1 = set()
for i in seqs:
    if len(i) > 1 and len(i) < 201:
        sq1.add(i)

# dist of peptide lengths of peptides in the rnage 2 to 200 aa:
#find_length_dist(sq1)

print(len(sq1), len(sq1)/len(seqs),"% of all unique sequences remain")



## convering to dict and saving
#aa_counter(sq1, vocab)
seqs = {"neuropeptide":list(set(sq1))}
with open("neuropedia_dct.pkl","wb") as f:
    pickle.dump(seqs,f)

neuropedia_seq_dct = {}
for k,v in seqs.items():
    for i in v:
        neuropedia_seq_dct[i] = set([k])

with open("neuropedia_seq_dct.pkl","wb") as f:
    pickle.dump(neuropedia_seq_dct,f)


######### Exp. validated 


#############################################################
#############################################################
#############################################################

### CancerPDD


bi = []
with open("cancerppd/anticancer.txt", "r", encoding='utf-8') as f:
    for i in f:
        bi.append(i.strip().split("\t")[1])

## remove sequences if they have letters not in my vocabulary
seqs = set()
bi = bi[1:]
for i in bi:
    if len(set(list(i.upper())).difference(set(vocab))) == 0:
        seqs.add(i.upper())
    else:
        removed_sequences.append(i.upper())


## counting unique peptides
print("number of unique peptides:", len(seqs))


## removing sequences if they are not in the size range 2-200 aa
sq1 = set()
for i in seqs:
    if len(i) > 1 and len(i) < 201:
        sq1.add(i)

# dist of peptide lengths of peptides in the rnage 2 to 200 aa:
#find_length_dist(sq1)


print(len(sq1), len(sq1)/len(seqs),"% of all unique sequences remain")


## convering to dict and saving
#aa_counter(sq1, vocab)
anticancer_seq_dct = {}
for i in sq1:
    anticancer_seq_dct[i] = set(["anticancer"])

with open("anticancer_seq_dct.pkl","wb") as f:
    pickle.dump(anticancer_seq_dct,f)



######### Exp. validated

#############################################################
#############################################################
#############################################################

bk = pd.read_csv("biodadpep/biodadpep_data.csv")
new_bk = bk.values
tmpseqs = new_bk[:,2]


## remove sequences if they have letters not in my vocabulary
diabseqs = set()
for i in tmpseqs:
    if len(set(list(i.upper())).difference(set(vocab))) == 0:
        diabseqs.add(i.upper())
    else:
        removed_sequences.append(i.upper())

## counting unique peptides
print("number of unique peptides:", len(diabseqs))


## removing sequences if they are not in the size range 2-200 aa
sq1 = set()
for i in diabseqs:
    if len(i) > 1 and len(i) < 201:
        sq1.add(i)

# dist of peptide lengths of peptides in the rnage 2 to 200 aa:
#find_length_dist(sq1)


print(len(sq1)/len(diabseqs),"% of all unique sequences remain")


## convering to dict and saving
antidiab = {}
for i in sq1:
    antidiab[i] = set(["antidiabetes"])

with open("diabetes_seq_dct.pkl","wb") as f:
    pickle.dump(antidiab,f)

#aa_counter(sq1, vocab)


#############################################################
#NeuroPeP
print("NeuroPeP")

sq = []
ids = []
with open("neuropep/all_fasta.fa", "r") as f:
    for i in f:
        if i[0] == ">":
            ids.append(i.strip())
        else:
            sq.append(i.strip())


## remove sequences if they have letters not in my vocabulary
nesq = set()
for i in sq:
    if len(set(list(i.upper())).difference(set(vocab))) == 0:
        nesq.add(i.upper())
    else:
        removed_sequences.append(i.upper())



## counting unique peptides
sq = list(set(nesq))
print("number of unique peptides:", len(sq))

lens = {}
for i in sq:
    ln = len(i)
    if ln in lens:
        lens[ln] += 1
    else:
        lens[ln] = 1

# dist of peptide lengths:
#find_length_dist(sq)
#plt.close()



## removing sequences if they are not in the size range 2-200 aa
sq1 = []
for i in sq:
    if len(i) > 1 and len(i) < 201:
        sq1.append(i)

# dist of peptide lengths of peptides in the rnage 2 to 200 aa:
#find_length_dist(sq1)

print(len(sq1)/len(sq),"% of all unique sequences remain")



## convering to dict and saving
nepep = {}
for i in sq1:
    nepep[i] = set(["neuropeptide"])


with open("neuropep_seq_dct.pkl","wb") as f:
    pickle.dump(nepep,f)

#aa_counter(sq1, vocab)


#############################################################
#############################################################
#############################################################


with open("removed_sequences.txt","w", encoding='utf-8') as f:
    for i in removed_sequences:
        f.write(i+"\n")



#############################################################
#############################################################
#############################################################

# taking all dicts with sequence and label info
alz = [camp_sq_data, lamp_seq_dct, apd3_seq_dct, satp_seq_dct,
dbaasp_seq_dct,biopep_seq_dct,peptidedb_seq_dct,neuropedia_seq_dct,
anticancer_seq_dct, antidiab, nepep]

# compiling into one dictionary
total = {}
for i in alz:
    for k,v in i.items():
        line = k.strip().upper()
        if len(line) <= 200 and len(line) > 1:              # size check 
            if len(set(list(line)).difference(vocab)) == 0: # vocab check
                
                if line in total:
                    total[line].update(set(v))
                elif line not in total:
                    total[line] = set(v)
            else:
                print(k)

# generating a set with all classes
all_classes = set()
for k,v in total.items():
    all_classes.update(v)


# generating a dict that can connect redundant class names with a single class name
decoder = {"insecticides":{'insectcidal','insectcides','insecticidal','insecticides',},
           
           "woundhealing":{"woundhealing", "woundheal"},
           
           "antioxidative":{"antioxidative", "antioxidant"},
           
           "cellcellsignaling":{"cellcellsignaling", "cellcellcomminication"},
           
           "drugdelivery":{"drugdelivery", "drugdeliveryvehicle"},
           
           "enzymeinhibitor":{"enzymeinhibitor", "enzyme_inhibitor"},
           
           "hemolytic":{"hemolytic", "haemolytic"},
           
           "antivirus":{"anitvirus", "antivirus"},
           
           "antibacterial":{"MRSA", "tuberculosis", "antibacterial","antibiofilm"},
           
           "antitoxin":{"antiendotoxin", "antitoxin"},
           
           "antiparasite":{"antiparasite", "antiprozoal"},
           
           "cytokines_growthfactors":{"chemotactic", "cytokines_growthfactors"},
           
           "antihypertensive":{"antihypertensive", "hypotensive"},
           
           "antidiabetes":{"antidiabetic", "antidiabetes"},
           
           "antifungal":{"anitifungal","antifungal"}
    }


# making a set wirh all class names
decer = set()
for k, v in decoder.items():
    decer.update(v)


# merging similar calsses into one class
total_classes = {}
for k,v in total.items():
    if len(k) > 200 or len(k) < 2:  # removing sequences longer than 200 
                                    # and shorther than 2
        print(k)
        continue
    for i in v:
        if i in decer:
            for kk,vv in decoder.items():    
                if i in vv:
                    if kk in total_classes:
                        total_classes[kk].add(k)
                    if kk not in total_classes:
                        total_classes[kk] = {k}
                        
        else:
            if i in total_classes:
                total_classes[i].add(k)
            if i not in total_classes:
                total_classes[i] = {k}

# printing item per class
items_per_key(total_classes)

# generating a new set with all classes
all_classes = set()
for k,v in total_classes.items():
    all_classes.update([k])
all_classes_lst = list(all_classes)
all_classes_lst.sort()


# Finding how classes overlap
dcter = find_overlapping_items(total_classes)    
dcter_lst = find_overlapping_items_lst(total_classes)



# Removing classes with fewer than 100 peptides and if number of unique sequences are fewer than 10
remover = []
for k,v in total_classes.items():
    tester = set()
    for kk,vv in total_classes.items():
        if k != kk:
            tester.update(total_classes[k].intersection(vv))
    if len(v) - len(tester) < 10:
        remover.append(k)
   
for k,v in total_classes.items():
    tmp = set()
    if len(v) < 100:
        if k not in remover:
            remover.append(k)

for i in remover:
    total_classes.pop(i)
print("\nremoved:")
for i in remover:
    print(i)




# generating a new set with all classes
all_classes = set()
for k,v in total_classes.items():
    all_classes.update([k])
all_classes_lst = list(all_classes)
all_classes_lst.sort()

# making dicts and list with classes and how the overlap with each other
dcter_lst = find_overlapping_items_lst(total_classes)
dcter = find_overlapping_items(total_classes)




## generating a heatmap that show how the classes overlap
def generate_heatmap(all_classes_lst,dcter_lst):
    matrix = np.zeros((len(all_classes_lst), len(all_classes_lst)))
    for j,i in enumerate(all_classes_lst):
        for ii in dcter_lst[i]:
            matrix[j][all_classes_lst.index(ii[1])] = ii[0] / dcter_lst[i][-1][0]
    return matrix

m = generate_heatmap(all_classes_lst,dcter_lst)
sns.clustermap(m, method="complete", row_cluster=True, col_cluster=True, yticklabels=all_classes_lst,xticklabels=all_classes_lst, metric="euclidean", figsize=(20,20))
plt.savefig("heatmap2.png")


# making dendrogram that show how the classes overlap
matrix = np.zeros((len(all_classes_lst), len(all_classes_lst)))
for j,i in enumerate(all_classes_lst):
    for ii in dcter_lst[i]:
        matrix[j][all_classes_lst.index(ii[1])] = ii[0] / dcter_lst[i][-1][0]

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
linked = linkage(matrix.T, 'complete', metric="euclidean")
plt.figure(figsize=(25, 20))
den = dendrogram(linked,
            orientation='top',
            labels=all_classes_lst,
            distance_sort='descending',
            leaf_rotation = 60,
            show_leaf_counts=True)
plt.savefig("usethissT2.svg")

linked = linkage(matrix, 'complete', metric="euclidean")
plt.figure(figsize=(25, 20))
den = dendrogram(linked,
            orientation='top',
            labels=all_classes_lst,
            distance_sort='descending',
            leaf_rotation = 60,
            show_leaf_counts=True)
plt.savefig("usethiss2.svg")


# saving the classes
with open("total_classes32.pkl","wb") as f:
    pickle.dump(total_classes,f)



# generating clades with classes
leaves = np.array(all_classes_lst)[leaves_list(linked)]
cls1 = leaves[:2]
cls2 = leaves[2:8]
cls3 = leaves[8:11]
cls4 = leaves[11:16]
cls5 = leaves[16:]

branches = [cls1,cls2,cls3,cls4,cls5]

with open("branches32.pkl","wb") as f:
    pickle.dump(branches,f)

tmp = []
for i in branches:
    tmp += list(i)
assert tmp == list(leaves)



#### Creating the final dict with sequences and associated classes
tot_seq= {}
for k,v in total_classes.items():
    for i in v:
        if i in tot_seq:
            tot_seq[i].add(k)
        if i not in tot_seq:
            tot_seq[i] = {k}

print(len(tot_seq))#, len(tot_seq_250))

with open("total_classes_seq32.pkl","wb") as f:
    pickle.dump(tot_seq,f)



# creating a dict with class-sets and associated seqeunces
# this will be used to divide the sequences into ten bins for 10-fold cc
counter_labs = {}
for k,v in tot_seq.items():
    tmp = list(v)
    tmp.sort()
    tmp1 = ""
    for i in tmp:
        tmp1 += i+" | "
    if tmp1 in counter_labs:
        counter_labs[tmp1].add(k)
    else:
        counter_labs[tmp1] = {k}



lster = []
for k,v in counter_labs.items():
    lster.append([len(v),k])
lster.sort()


# function for binning up peptides
def binup_data(dct):
    
    # dividing seqeunces from class-sets into ten bins
    bins = {}
    for k,v in dct.items():     
        if len(v) < 10:
            continue
        tmp = []
        div = int(len(v)/10)
        V = list(v)
        random.shuffle(V)
        for i in range(10):
            if i == 9:
                tmp.append(V[div*(i):])
            else:
                tmp.append(V[div*(i):div*(i+1)])
        bins[k] = tmp
        
    # evening out sizes of bins
    for k,v in bins.items():
        diff = len(v[-1]) - len(v[0])
        if diff > 1:
            d1 = diff-1
            ad = v[-1][:d1]
            v[-1] = v[-1][d1:]
            for i in range(d1):
                v[i].append(ad[i])
    
    # printing data
    for k,v in bins.items():
        t0 = 0
        for i in v:
            t0 += len(i)
        print(k,len(v), len(dct[k]), t0)
        assert len(dct[k]) == t0
        [print(len(i),sep=" ", end=' ') for i in v]
        print("\n")
    
    for k,v in bins.items():
        for i in range(len(v)):
            for ii in range(len(v)):
                if i != ii:
                    if len(set(v[i]).intersection(v[ii])) > 0:
                        print(k, i, ii)
                    
    # binning class-sets smaller than ten at random
    for k,v in dct.items():
        if len(v) < 10:
            ntmp = []
            tmp = list(v)
            for i in range(10):
                try:
                    ntmp.append([tmp[i]])
                except:
                    ntmp.append([])
            m = np.arange(10)
            np.random.shuffle(m)
            ntmp = [ntmp[i] for i in m]
            bins[k] = ntmp
    
    return bins

binz = binup_data(counter_labs)


###
#checking that all sequences are there
ttt = set()
for k,v in binz.items():
    tt = 0
    for i in v:
        tt += len(i)
        ttt.update(i)
    assert tt == len(counter_labs[k])
assert len(ttt) == len(tot_seq)
print(len(ttt))
 
# saving the data
with open("total_classes_seq_bins32.pkl","wb") as f:
    pickle.dump(binz,f)


################################################################



# making barplots of peptide length distributions of classes
for k,v in total_classes.items():
    find_length_dist(k, v)
    
# making barplots of peptide amino acid distribution of classes
for k,v in total_classes.items():
    aa_counter(k, v, vocab)



# making a table with class and database overlaps
databases = ["CAMP3","LAMP2", "APD3", "SATPdb", "DBAASP", "BIOPEP-UWM", "PeptideDB", "NeuroPedia", "CancerPDD", "BioDADPep", "NeuroPep"] 
#databases.sort()
dat = []

tokey = list(total_classes.keys())
tokey.sort()
for k in tokey:
    print("\n",k)
    tmp = [k]
    for j,i in enumerate(alz):
        tmp.append(len(set(list(i.keys())).intersection(total_classes[k])))
        print(databases[j], len(set(list(i.keys())).intersection(total_classes[k])), end=" ", sep=" ")
    dat.append(tmp+[len(total_classes[k])])
    
dat = pd.DataFrame(dat)
dat.columns = ["Classes"] + databases + ["Total class size"]
dat.to_excel("table_class_db_overlap2.xlsx")




