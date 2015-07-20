#! /bin/bash

# Iterate / Call ExtractSNP.py
bam_dir="/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/EM_Clonal_Abundance/"
result_dir="/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/EM_Clonal_Abundance/"
PATH_TO_HG19REF="/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/EM_Clonal_Abundance/hs37d5.fa"
PATH_TO_SAMTOOLS="/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/samtools-1.1/samtools"
PATH_TO_BCFTOOLS="/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/bcftools-1.1/bcftools"
PATH_TO_SNPSIFT="/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/snpEff/SnpSift.jar"
mutation="DEL" # type of mutation, UPD or DEL (monosomy)
chr2use="5" # chromosome to use
position_range="137528564,153809148" # SNP position range "144707068,148942012"
min_read="" # any total reads below this number is filtered out, 10

# Assuming these files are sorted.
tum_bam_list=("D110RACXX_3_1ss_.sam.bam_sorted_only_mapped.bam") # separated by space
norm_bam_list=("D110RACXX_4_1ss_.sam.bam_sorted_only_mapped.bam") # separated by space
sample_list=("D110RACXX") # separated by space

# echo ${#tum_bam_list[@]}
# echo ${#norm_bam_list[@]}
# echo ${#sample_list[@]}
# exit 0

# Find script's own path; export some global variables;
PATH_INSTALL="$(dirname $(readlink -f ${BASH_SOURCE[0]}))"
export PATH_TO_HG19REF
export PATH_TO_SAMTOOLS
export PATH_TO_BCFTOOLS
export PATH_TO_SNPSIFT

# echo $PATH_TO_HG19REF
# echo $PATH_TO_SAMTOOLS
# echo $PATH_TO_BCFTOOLS
# echo $PATH_TO_SNPSIFT
# exit 0

# Record list of files to be used in EM
shared_snp_count_list=""

# Parse options for ExtractSNP
if [ -n "$chr2use" ]; then chr2use=" -r "${chr2use}; fi

# Call SNPs
for i in "${!sample_list[@]}"; do
	norm=${bam_dir}/${norm_bam_list[$i]}
	tum=${bam_dir}/${tum_bam_list[$i]}
	rst=${result_dir}/${sample_list[$i]}/

	shared_snp_count_list+="${rst}/shared_snp_count.txt "

	#Before run, check if this sample is already processed
	if [ -e ${rst}/shared_snp_count.txt ]; then
		continue
	fi

	# Call
	echo "python ${PATH_INSTALL}/ExtractSNP.py $chr2use ${PATH_TO_HG19REF} ${norm} ${tum} ${rst}"
	python ${PATH_INSTALL}/ExtractSNP.py $chr2use ${PATH_TO_HG19REF} ${norm} ${tum} ${rst}
done

# Parse Options for EM
if [ -n "$mutation" ]; then mutation="-m "${mutation}; fi
if [ -n "$position_range" ]; then position_range=" -p "${position_range}; fi
if [ -n "$min_read" ]; then min_read=" -N "${min_read}; fi
output=" -o "${result_dir}"/theta_results"`date +'_%Y.%m.%d.%H.%M.%S'`".tsv"
subject_id=" -i "$(IFS=$','; echo "${sample_list[*]}")
options=${mutation}${chr2use}${position_range}${min_read}${subject_id}${output}


# Estimate abundance with EM
echo "python ${PATH_INSTALL}/EM.py ${options} ${shared_snp_count_list}"
python ${PATH_INSTALL}/EM.py ${options} ${shared_snp_count_list}
