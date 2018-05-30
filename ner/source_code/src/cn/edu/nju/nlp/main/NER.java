package cn.edu.nju.nlp.main;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

public class NER {
	
	public static int TRAIN_MODE = 0;
	public static int PRED_MODE = 1;
	public static int EVAL_MODE = 2;
	
	public static Properties property;

	
	public static void main(String[] args) {
		
		init();
				
		String trainFile = property.getProperty("TRAIN_FILES");
		String testFile = property.getProperty("TEST_FILES");
		String paramsFile = property.getProperty("PARAM_FILE");
		String resultFile = property.getProperty("RESULT_FILE");
		String goldFile = property.getProperty("GOLD_FILE");
		String locDictFile = property.getProperty("LOCDICT_FILE");
		String shortLocDictFile = property.getProperty("SHORTLOCDICT_FILE");
		
		String NER_Label = property.getProperty("NER_LABEL");
		String[] NER_Labels = NER_Label.split("/");
		String OTHER_Label = property.getProperty("OTHER_LABEL");
		String[] ALL_Labels = new String[4*NER_Labels.length+1];
		ALL_Labels[0] = OTHER_Label;
		for(int i = 1; i <= NER_Labels.length; ++i){
			ALL_Labels[(i<<2)-3] = "B_"+NER_Labels[i-1];
			ALL_Labels[(i<<2)-2] = "I_"+NER_Labels[i-1];
			ALL_Labels[(i<<2)-1] = "E_"+NER_Labels[i-1];
			ALL_Labels[(i<<2)-0] = "S_"+NER_Labels[i-1];
		}
		
		boolean use_postag = Boolean.parseBoolean(property.getProperty("USE_POSTAG"));
		boolean train_flag = Boolean.parseBoolean(property.getProperty("IF_TRAIN"));
		boolean pred_flag = Boolean.parseBoolean(property.getProperty("IF_PRED"));
		boolean eval_flag = Boolean.parseBoolean(property.getProperty("IF_EVAL"));

		if(train_flag) NERTrain(trainFile, paramsFile, locDictFile, shortLocDictFile, use_postag, ALL_Labels);
		if(pred_flag) NERPred(testFile, paramsFile, locDictFile, shortLocDictFile, resultFile, use_postag, ALL_Labels, OTHER_Label);
		if(eval_flag){
			Util u = new Util();
			u.evaluation(goldFile, resultFile, use_postag, NER_Labels);			
		}
	}

	private static void init(){
		property = new Properties();
		FileReader fr;
		try {
			fr = new FileReader("model.properties");
			property.load(fr);
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void NERTrain(String trainFile, String paramsFile, String dictFile, String shortDictFile, boolean use_postag, String[] all_labels) {
		System.out.println("***************************Train*****************************");

		int beamSize = 16, mode = TRAIN_MODE, iteration = 10;
		Map<String, double[]> params = new HashMap<>();

		Beam_search b = new Beam_search();
		Util u = new Util();

		// 将transFile文件里的内容以行的形式返回
		List<String> sentenceList = u.loadFile(trainFile);
		Set<String> locDict = u.loadDicts(dictFile);
		Set<String> shortLocDict = u.loadDicts(shortDictFile);

		long start = System.currentTimeMillis();
		for (int i = 0; i < iteration; ++i) {
			System.out.println("iteration: " + i + " Time: " + (System.currentTimeMillis() - start) / 1000 + "s");
			List<Term> temp = new ArrayList<>();
			for (int j = 0; j < sentenceList.size(); ++j) {
				String sentence = sentenceList.get(j);
				temp = b.runBeamSearch(sentence, beamSize, params, locDict, shortLocDict, i * sentenceList.size() + j,
						mode, use_postag, all_labels);
			}
			System.out.println("parameter size is " + params.size());
			temp.clear();
		}

		int length = sentenceList.size();
		sentenceList.clear();

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(paramsFile), "utf-8"));
			for (Map.Entry<String, double[]> entry : params.entrySet()) {
				if (entry.getValue()[0] != 0) // 删除值为0的特征
					bw.write(entry.getKey() + '\t' + entry.getValue()[0] / (length * iteration) + '\n');
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		finally {
			try {
				bw.flush();
				bw.close();
			}
			catch (IOException e) {
				e.printStackTrace();
			}
		}

		System.out.println("Training done! Time: "+(System.currentTimeMillis() - start) / 1000 + "s");
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println();
	}

	private static void NERPred(String testFile, String paramsFile, String dictFile, String shortDictFile, String resultFile, boolean use_postag, String[] all_labels, String OTHER_Label) {
		System.out.println("**************************Predict****************************");
		
		int beamSize = 16, mode = PRED_MODE, now = 0;
		Util u = new Util();
		Map<String, double[]> params = u.loadParams(paramsFile);
		Set<String> locDict = u.loadDicts(dictFile);
		Set<String> shortLocDict = u.loadDicts(shortDictFile);

		List<String> sentenceList = u.loadFile(testFile);

		Beam_search b = new Beam_search();
		List<List<String[]>> result = new ArrayList<>();

		long start = System.currentTimeMillis();
		for (int i = 0; i < sentenceList.size(); ++i) {
			List<String[]> wordList = u.genWordPosLabel(sentenceList.get(i), mode, use_postag);
			List<Term> predictTerm = b.runBeamSearch(sentenceList.get(i), beamSize, params, locDict, shortLocDict, now,
					mode, use_postag, all_labels);
			for (int j = 0; j < predictTerm.size(); ++j) {
				wordList.get(j)[2] = predictTerm.get(j).label;
			}
			if(i > 0 && i % 5000 == 0) System.out.println(i+" sentences predict done!");
			result.add(u.simplifyLabel(wordList, OTHER_Label));
		}

		sentenceList.clear();
		params.clear();

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(resultFile), "utf-8"));
			for (int i = 0; i < result.size(); ++i) {
				for (int j = 0; j < result.get(i).size(); ++j) {
					bw.write(result.get(i).get(j)[0] + "/" + result.get(i).get(j)[2] + " ");
				}
				bw.write("\n");
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			try {
				bw.flush();
				bw.close();
			}
			catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		System.out.println("Predicting done! Time: "+(System.currentTimeMillis() - start) / 1000 + "s");
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println();
	}
}
