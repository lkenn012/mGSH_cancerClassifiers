### code to get FASTA sequences for a list of genes using the ensembl API

# import modules
import requests

# List of gene symbols
gene_symbols = ["TP53", "BRCA1", "EGFR"] 
gene_symbols = ["OGDH", "AADAT", "IDH2", "D2HGDH", "PHYH", "GPT2", "DLST", "ADHFE1", "IDH1", "OGDHL", "GOT1", "TAT", "GOT2", "L2HGDH", "MRPS36"]
gene_symbols = ["ENSG00000105953", "ENSG00000109576", "ENSG00000182054"] # , "ENSG00000180902", "ENSG00000107537", "ENSG00000166123", "ENSG00000119689", "ENSG00000147576", "ENSG00000138413", "ENSG00000197444", "ENSG00000120053", "ENSG00000198650", "ENSG00000125166", "ENSG00000087299", "ENSG00000134056"] 

# define function to get the canonical transcript ID for a given Ensembl gene ID using the ensembl API
def get_canonicalTranscript(gene_id, version=False):
    url = f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1"
    response = requests.get(url, headers={"Content-Type": "application/json"})
    data = response.json()

    # if id is not found data will be returned with 'error' key only
    if "error" in data:
        print(f'error:\n {data["error"]}')
        return None

    # By default, IDs are returned with version IDs (ENSG000123.1) but these need to be removed when getting protein seqs, for e.g.
    if version:
        return data["canonical_transcript"]

    else:
        id_version = data["canonical_transcript"].split(".")    # split into transcript ID and version ID
        return id_version[0]    # return ID only

# define
def get_proteinSeq(transcript_id):
    url = f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=protein"
    response = requests.get(url, headers={"Content-Type": "text/plain"})
    return response.text

# Example usage
def main():
    gene_id = "ENSG00000133475"  # Replace with your actual ENSG gene ID
    canonical_transcript_id = get_canonicalTranscript(gene_id)
    print(f"Canonical Transcript ID: {canonical_transcript_id}")

    protein_sequence = get_proteinSeq(canonical_transcript_id)   # [:-2] removes the version number from the ID
    print(f"Protein Sequence:\n{protein_sequence}")

# main()
