DATASET_PATH='testdata_various_noises.h5'
CASE='various_noises'

# Just some colors to lines on the terminal.
RED=`tput setaf 1`
ORANGE=`tput setaf 3`
DEFAULT=`tput sgr0`

################### Create dataset to test the networks ###################
if [ -f $DATASET_PATH ]; then
	echo "${RED}Testing dataset already exists! Ensure it doesnt affect the testing!${DEFAULT}"
else
	echo "${RED}Creating test dataset ...{DEFAULT}"
	python3 create_statistical_data.py --case $CASE --name $DATASET_PATH
fi
echo -e ''

################### Tests of Mask-PointNetLK, Mask-ICP, Mask-DCP ###################
MASKNET='True'
echo -e "${RED}Using MaskNet: "$MASKNET"\n${DEFAULT}"
for REG_ALGO in 'pointnetlk' 'icp' 'dcp'
do
	RESULTS_DIR='results_masknet_'$reg_algorithm'_various_noises'
	echo -e "${ORANGE}Registration Algorithm Getting Tested: "$REG_ALGO"\n${DEFAULT}"

	# Test the network for different angles from 0 to 90 degrees.
	for group_no in 0 0.02 0.04 0.06 0.1 0.15 0.2
	do
		GROUP_NAME='noise_'$group_no
		echo "${ORANGE}Value of Initial Misalignment: "$group_no"${DEFAULT}"
		python3 evaluate_stats.py --dataset_path $DATASET_PATH --results_dir $RESULTS_DIR --group_name $GROUP_NAME --masknet $MASKNET --reg_algorithm $REG_ALGO
	done
done

################### Tests of PointNetLK, ICP, DCP ###################
MASKNET='False'
echo -e "${RED}Using MaskNet: "$MASKNET"\n${DEFAULT}"
for REG_ALGO in 'pointnetlk' 'icp' 'dcp'
do
	RESULTS_DIR='results_'$reg_algorithm'_various_noises'
	echo -e "${ORANGE}Registration Algorithm Getting Tested: "$REG_ALGO"\n${DEFAULT}"

	# Test the network for different angles from 0 to 90 degrees.
	for group_no in 0 0.02 0.04 0.06 0.1 0.15 0.2
	do
		GROUP_NAME='noise_'$group_no
		echo "${ORANGE}Value of Initial Misalignment: "$group_no"${DEFAULT}"
		python3 evaluate_stats.py --dataset_path $DATASET_PATH --results_dir $RESULTS_DIR --group_name $GROUP_NAME --masknet $MASKNET --reg_algorithm $REG_ALGO
	done
done


################### Create Plots ###################
CASE='noise'
for REG_ALGO in 'pointnetlk' 'icp' 'dcp'
do
	for METRIC in 'rotation_error' 'translation_error'
	do
		python3 plot_results.py --case $CASE --reg_algorithm $REG_ALGO --metric $METRIC
	done
done