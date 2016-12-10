python CRFNER.py

echo "CRF trained, dumped model at linear-chain-crf.model.pickle"

pdtxts=results/crfprediction/nolabel/predictedtxts/
pdanns=results/crfprediction/nolabel/predictedanns/

rm -rf $pdtxts
rm -rf $pdanns

mkdir -p $pdtxts
mkdir -p $pdanns

for f in ./data/conllformat/nolabel/test/withtokenindex/*withtokenindex.txt
do
  echo "classifying $f with CRF"
  python ClassifyWithCRF.py $f $pdtxts
done

for f in  $pdtxts/*-crfprediction.txt 
do
  echo "creating ann file for predicted $f"
  python ConvertCoNLLtoANN.py $f $pdanns
done

folder_gold=data/semevaltrainingdata/anns/test
folder_predicted=$pdanns


python eval.py $folder_gold $folder_predicted

