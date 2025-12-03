import polars as pl
import pysam
import re
import requests

AAS = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Glu": "E",
    "Gln": "Q",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
    "Sec": "U",
    "Pyl": "O",
}


def get_dna_window(row: dict, fa: pysam.FastaFile, flank: int = 512):
    chrom = str(row["Chromosome"])          # e.g. '15'
    pos = int(row["PositionVCF"])           # 1-based
    ref = str(row["ReferenceAlleleVCF"])    # 'G'
    alt = str(row["AlternateAlleleVCF"])    # 'A'

    start = pos - flank
    end = pos + flank

    # pysam: 0-based, half-open
    wt_seq = fa.fetch(chrom, start - 1, end)

    # sanity check
    ref_base = fa.fetch(chrom, pos - 1, pos).upper()
    if ref_base != ref:
        raise ValueError(
            f"Ref mismatch at {chrom}:{pos}: fasta={ref_base}, ClinVar={ref}"
        )

    offset = flank  # position of variant within the window
    alt_seq = wt_seq[:offset] + alt + wt_seq[offset + 1:]

    return wt_seq, alt_seq


def get_uniprot_acc(other_ids: str | None):
    m = re.search(r"UniProtKB:([A-Z0-9]+)", str(other_ids))
    return m.group(1) if m else None


def fetch_uniprot_seq(acc: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">>") and not l.startswith(">"))


def parse_protein_hgvs(name: str):
    # handles p.Gly1046Arg style
    m = re.search(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})", str(name))
    if not m:
        raise ValueError(f"Cannot parse protein HGVS from {name}")
    ref3, pos, alt3 = m.groups()
    pos = int(pos)
    ref1 = AAS[ref3]
    alt1 = AAS[alt3]
    return pos, ref1, alt1


def get_protein_wt_mut(row: dict):
    acc = get_uniprot_acc(row["OtherIDs"])
    if acc is None:
        raise ValueError("No UniProt accession found in OtherIDs")

    prot_wt = fetch_uniprot_seq(acc)
    pos, ref_aa, alt_aa = parse_protein_hgvs(row["Name"])

    if prot_wt[pos - 1] != ref_aa:
        raise ValueError(
            f"Ref AA mismatch: protein has {prot_wt[pos - 1]} at {pos}, HGVS says {ref_aa}"
        )

    prot_mut = prot_wt[: pos - 1] + alt_aa + prot_wt[pos:]
    return prot_wt, prot_mut


def process_variant_row(row: dict, fa: pysam.FastaFile, flank: int = 512):
    """
    Try to build everything we need for one SNV row.
    Return a dict if successful, or None if anything fails.
    """
    try:
        # 1) DNA windows (wt + alt)
        wt_dna, alt_dna = get_dna_window(row, fa=fa, flank=flank)

        # 2) Protein sequences (wt + mutant)
        prot_wt, prot_mut = get_protein_wt_mut(row)

        # 3) Label from ClinSigSimple (already 0/1)
        label = int(row["ClinSigSimple"])

    except Exception:
        # Any failure → skip this variant
        return None

    return {
        "variant_id": int(row["VariationID"]),
        "chrom": str(row["Chromosome"]),
        "pos": int(row["PositionVCF"]),
        "ref": str(row["ReferenceAlleleVCF"]),
        "alt": str(row["AlternateAlleleVCF"]),
        "gene_symbol": str(row["GeneSymbol"]),
        "wt_dna": wt_dna,
        "alt_dna": alt_dna,
        "prot_wt": prot_wt,
        "prot_mut": prot_mut,
        "label": label,
    }


def build_snv_dataset(df: pl.DataFrame, fa: pysam.FastaFile, flank: int = 512) -> pl.DataFrame:
    """
    Go over all rows, keep only SNVs where both DNA + protein processing work.
    Returns a clean Polars DataFrame ready for embedding.
    """
    records = []

    # iter_rows(named=True) gives dict-like rows
    for row in df.iter_rows(named=True):
        # Only single nucleotide variants (df is probably already filtered, but keep this guard)
        if row["Type"] != "single nucleotide variant":
            continue

        rec = process_variant_row(row, fa=fa, flank=flank)
        if rec is not None:
            records.append(rec)

    if not records:
        return pl.DataFrame()  # empty

    return pl.DataFrame(records)


def main():
    # read TSV via Polars
    print("reading variant_summary.txt with Polars")
    snv = (
        pl.read_csv(
            "data/variant_summary.txt",
            separator="\t",
            ignore_errors=True,  # in case of some weird lines
        )
        .filter(pl.col("Type") == "single nucleotide variant")
    )

    print("loading fasta")
    fa = pysam.FastaFile("data/GRCh38.fa")  # or GRCh37.fa, depending on your assembly

    print("building dataset (DNA + protein)")
    snv_dna_aa = build_snv_dataset(snv, fa, flank=512)

    print("df shape:", snv_dna_aa.shape)

    print("saving to parquet with snappy compression")
    snv_dna_aa.write_parquet(
        "snvs.parquet",
        compression="snappy",
    )


if __name__ == "__main__":
    main()