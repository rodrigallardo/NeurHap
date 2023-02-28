# %%
import numpy as np
import pandas as pd


# %%
######### Utils ##########
def base_encode(base):
    # Function to stablish encoding for the bases, where:
    # A -> 1
    # C -> 2
    # G -> 3
    # T -> 4
    mapping = {"A": 1, "C": 2, "G": 3, "T": 4}
    return mapping[base] if base in mapping.keys() else print("Non-base in vcf")


def get_base(pair_reads_row, seq_column_name, read_col_name, snv_pos):
    # This should take a position value in a read and get the corresponding base
    # get relevant seq from pair reads column: pair_reads_row[seq_column]
    # get relative index for snv in seq: [snv_pos - pair_reads_row[read_col_name]]
    base = pair_reads_row[seq_column_name][snv_pos -
                                           pair_reads_row[read_col_name]]
    if base in "ACGT":
        return base_encode(base)
    else:
        return 0


# %%
##########################
# Set filename values to the files to be analyzed
vcf_filename = "U7_pass_chr22_snps.vcf"
sam_filename = "U7_sorted_chr22.sam"  # This sorted sam is sorted by position
# %%
# Create a df from a vcf file
# The vcf file should only contain snp positions (no indels, dups, etc)
# you can use bcftools to filter your vcf

with open(vcf_filename, "r") as f:
    lines = f.readlines()
    chrom_index = [i for i, line in enumerate(
        lines) if line.strip().startswith("#CHROM")]
    data = lines[chrom_index[0]:]
    header = data[0].strip().split("\t")
    informations = [d.strip().split("\t") for d in data[1:]]

vcf_df = pd.DataFrame(informations, columns=header)

# Optimize dtypes, TODO: check max/min values of positions to asign memory automatically
vcf_df["#CHROM"] = vcf_df["#CHROM"].astype("category")
vcf_df["POS"] = vcf_df["POS"].astype("uint32")
vcf_df["ID"] = vcf_df["ID"].astype("category")
vcf_df["QUAL"] = vcf_df["QUAL"].astype("float32")
vcf_df["FILTER"] = vcf_df["FILTER"].astype("category")

# %%
# Create a df from a sam file
sam_df = pd.read_csv(
    sam_filename,
    sep="\t",
    names=["QNAME", "FLAG", "RNAME", "POS", "MAPQ", "CIGAR", "RNEXT", "PNEXT",
           "TLEN", "SEQ", "QUAL", "opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7"],
)

# Apply quality filters:
# filter sam to rows with MAPQ > min_mapq
min_mapq = 60
sam_df = sam_df[sam_df["MAPQ"] >= min_mapq]
# We want to keep the reads that matched to the reference
sam_df = sam_df[sam_df["CIGAR"].str.contains("M")]
sam_df["READ_LENGTH"] = sam_df["CIGAR"].str.split("M", expand=True)[0]
sam_df = sam_df[~sam_df["READ_LENGTH"].str.contains("[a-zA-Z]")]
# We want to keep only the reads with the following FLAG values: {99, 163,  83, 147}
sam_df = sam_df[sam_df["FLAG"].isin([99, 163,  83, 147])]
# Filter out reads that don't have pairs
sam_df = sam_df[sam_df["QNAME"] == sam_df["QNAME"]]
# Keep only cols to be used
sam_df = sam_df.drop(columns=["RNAME", "MAPQ", "CIGAR", "RNEXT",
                              "QUAL", "opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7"],)

# Optimize dtypes, TODO: check max/min values of positions to asign memory automatically
sam_df["FLAG"] = sam_df["FLAG"].astype("uint8")
sam_df["POS"] = sam_df["POS"].astype("uint32")
sam_df["TLEN"] = sam_df["TLEN"].astype("int32")
sam_df["READ_LENGTH"] = sam_df["READ_LENGTH"].astype("uint16")
# %%
########################
### MATRIX CREATION ####
########################
# We want to have a SNP matrix were each row represents a reads pair
# and the columns represent the number of SNPs found in the vcf

# Set constants
RECONSTRUCTION_START = sam_df["POS"].min()
RECONSTRUCTION_END = sam_df["POS"].min() + 10000
MIN_READ_LENGTH = 70


# Filter df so we get the reads that match our reconstrucction positions
# TODO we could keep the segments of reads that start before the reconstruction_start but
# end after the reconstruction start
filtered_vcf_df = vcf_df[(vcf_df["POS"] >= RECONSTRUCTION_START) & (
    vcf_df["POS"] <= RECONSTRUCTION_END)]
# Get SNP positions:
SNPs = filtered_vcf_df["POS"].values

filtered_sam_df = sam_df[(sam_df["POS"] >= RECONSTRUCTION_START) & (
    sam_df["POS"] <= RECONSTRUCTION_END) & (sam_df["READ_LENGTH"] >= MIN_READ_LENGTH)]

# Create pair end read df
# We want to create a df that has the data of both read pair reads in one row
# Find first instance of the reads, this will be the first read in position order
# because the sam file is ordered by sequence
pair_reads_df = filtered_sam_df.drop_duplicates(subset="QNAME", keep="first")
complement_reads = filtered_sam_df[filtered_sam_df.duplicated(
    subset="QNAME", keep="first") == True]
# Merge df
pair_reads_df = pair_reads_df.merge(right=complement_reads, on="QNAME")

# Now we need to create the 0s SNP matrix
# Get the size params of the matrix
total_snv = len(filtered_vcf_df)
total_reads = len(pair_reads_df)

# We create the emtpy snv matrix using dtype as np.int8 as we will only use the following
# set of numbers for the matrix values: {0, 1, 2, 3, 4}
snv_matrix = np.zeros((total_reads, total_snv), dtype=np.int8)
# %%
# We need to fill the snv_matrix, to do so we will:
# 1. Iter through each array column so we can fill it SNP wise (column wise)
# To do so we need to determine what reads have info on the SNP's position, and add that info to
# the matrix
# -> find the POS value of the SNP
# -> for each read_pair that contains the POS, find if
# its inside the first read of the pair,
# inside the second read of the pair
# in the gap between de reads -> the original ExtractMatrix added 0s if this happened so we will too
# Depending where we find the SNP in the read, if we change the matrix's value or not
# Add the one hot encoded version of the correct base
for snv in range(0, total_snv):
    pos = SNPs[snv]
    for read in range(0, total_reads):
        # State different possible scenarios
        # a. snv is contained in the first read of the pair
        if (pair_reads_df.loc[read, "POS_x"] <= pos) & (pos < (pair_reads_df.loc[read, "POS_x"]+pair_reads_df.loc[read, "READ_LENGTH_x"])):
            snv_matrix[read, snv] = get_base(
                pair_reads_df.loc[read, :], "SEQ_x", "POS_x", pos)
        # b. snv is contained in the second read of the pair
        elif (pair_reads_df.loc[read, "POS_y"] <= pos) & (pos < (pair_reads_df.loc[read, "POS_y"]+pair_reads_df.loc[read, "READ_LENGTH_y"])):
            snv_matrix[read, snv] = get_base(
                pair_reads_df.iloc[read, :], "SEQ_y", "POS_y", pos)
        # c. snv isn't in the read pair
        else:
            pass

# %%
# Save snv_matrix to txt file
out_filename = "out.txt"

with open(out_filename, 'wb') as f:
    np.savetxt(f, snv_matrix, delimiter=' ', newline='\n',
               header='', footer='', comments='# ')
# %%
