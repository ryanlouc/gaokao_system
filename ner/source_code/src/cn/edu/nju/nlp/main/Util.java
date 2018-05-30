package cn.edu.nju.nlp.main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Util {
	// 加载文件
	List<String> loadFile(String filename) {
		File file = new File(filename);
		List<String> data = new ArrayList<>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String line = br.readLine();
			if (line.startsWith("\uFEFF")) {
				line = line.substring(1);
			}
			while (line != null) {
				data.add(line.trim());
				line = br.readLine();
			}
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			try {
				br.close();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		System.out.println("Data file <" + filename + "> loads done! Data size is " + data.size());
		return data;
	}

	// 加载参数
	Map<String, double[]> loadParams(String paramsFile) {
		File file = new File(paramsFile);
		Map<String, double[]> params = new HashMap<>();

		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String s = null;
			while ((s = br.readLine()) != null) {
				String[] str = s.split("\t");
				double[] tempDouble = { Double.parseDouble(str[1]), 0.0, 0.0 };
				params.put(str[0], tempDouble);
			}
		}
		catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			try {
				br.close();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		System.out.println("Parameter file <" + paramsFile + "> loads done! Parameter size is " + params.size());
		return params;
	}

	// 加载词典
	Set<String> loadDicts(String dictFile) {
		File file = new File(dictFile);
		Set<String> dicts = new HashSet<>();

		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String s = null;
			while ((s = br.readLine()) != null) {
				dicts.add(s.trim());
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		finally {
			try {
				br.close();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		System.out.println("Dictionary file <" + dictFile + "> loads done! Dictionary size is " + dicts.size());
		return dicts;
	}
	
	// 将训练文件的原始标签转化为BIESO标签
	void transformLabel(List<String[]> wordList){
		for(int i = 0; i < wordList.size(); ++i){
			if(!wordList.get(i)[2].equals("OTHER")){
				String[] temp = wordList.get(i);
				int j = i+1;
				while(j < wordList.size() && wordList.get(j)[2].equals(temp[2])){
					j++;
				}
				if(i == j-1){
					wordList.get(i)[2] = "S_"+temp[2];
				}
				else{
					wordList.get(i)[2] = "B_"+temp[2];
					i++;
					while(i < j){
						if(i == j-1){
							wordList.get(i)[2] = "E_"+wordList.get(i)[2];
						}
						else{
							wordList.get(i)[2] = "I_"+wordList.get(i)[2];
						}
						i++;
					}
					i = j-1;
				}
			}
		}
	}

	/**
	 * 对一句话生成(词/词性/标签)序列
	 * 
	 * @param mode
	 *            0表示训练，1表示测试
	 */
	List<String[]> genWordPosLabel(String sentence, int mode, boolean use_postag) {
		List<String[]> wordList = new ArrayList<>();
		String[] wordLabels = sentence.split(" ");
		for (String wordLabel : wordLabels) {
			String[] temp = wordLabel.split("/");
			String[] t = {temp[0], null, null };
			if(use_postag && (mode == NER.TRAIN_MODE || mode == NER.PRED_MODE)){
				t[1] = temp[1];
			}
			if (mode == NER.TRAIN_MODE || mode == NER.EVAL_MODE) {
				t[2] = temp[temp.length-1];
			}
			wordList.add(t);
		}
		if(mode == NER.TRAIN_MODE) transformLabel(wordList);
		return wordList;
	}
	
	/**
	 * Term长度为sentence中词的个数+1
	 * 
	 * @param wordList
	 * @return
	 */
	List<Term> genTerm(List<String[]> wordList) {
		List<Term> termList = new ArrayList<>();
		termList.add(new Term(0.0, -1, null, null));
		for (int i = 0; i < wordList.size(); ++i) {
			Term temp = new Term(0.0, i, wordList.get(i)[2], termList.get(termList.size() - 1));
			termList.add(temp);
		}
		return termList;
	}

	// 简化标签
	List<String[]> simplifyLabel(List<String[]> wordLabel, String OTHER_Label) {
		for (int i = wordLabel.size() - 1; i >= 0; --i) {
			if (!wordLabel.get(i)[2].equals(OTHER_Label)){
				wordLabel.get(i)[2] = wordLabel.get(i)[2].substring(2);
			}
		}
		return wordLabel;
	}

	void evaluation(String goldFile, String resultFile, boolean use_postag, String[] NER_Labels) {
		System.out.println("*************************Evaluation**************************");

		Map<String, double[]> evaluation = new HashMap<>();

		for (String temp : NER_Labels) {
			double[] array = { 0.0, 0.0, 0.0 };
			evaluation.put(temp, array);
		}

		Util u = new Util();
		List<String> sentenceTest = u.loadFile(goldFile);
		List<String> sentenceResult = u.loadFile(resultFile);
		for (int i = 0; i < sentenceTest.size(); ++i) {
			List<String[]> tempTest = u.genWordPosLabel(sentenceTest.get(i), NER.EVAL_MODE, use_postag);
			List<String[]> tempResult = u.genWordPosLabel(sentenceResult.get(i), NER.EVAL_MODE, use_postag);
			for (int j = 0; j < tempTest.size(); ++j) {
//				System.out.println(tempTest.get(j)[tempTest.get(j).length - 1]);
//				System.out.println(tempResult.get(j)[tempResult.get(j).length - 1]);
//				System.out.println();
				if (tempTest.get(j)[tempTest.get(j).length - 1].equals(tempResult.get(j)[tempResult.get(j).length - 1]) &&
					evaluation.containsKey(tempResult.get(j)[tempResult.get(j).length - 1])) {
					evaluation.get(tempTest.get(j)[tempTest.get(j).length - 1])[0] += 1.0;
				}

				if (evaluation.containsKey(tempResult.get(j)[tempResult.get(j).length - 1])) {
					evaluation.get(tempResult.get(j)[tempResult.get(j).length - 1])[1] += 1.0;
				}

				if (evaluation.containsKey(tempTest.get(j)[tempTest.get(j).length - 1])) {
					evaluation.get(tempTest.get(j)[tempTest.get(j).length - 1])[2] += 1.0;
				}
			}
		}
		
		DecimalFormat df = new DecimalFormat("0.00");
		for (Map.Entry<String, double[]> eachEval : evaluation.entrySet()) {
			double precision = eachEval.getValue()[0] / eachEval.getValue()[1];
			double recall = eachEval.getValue()[0] / eachEval.getValue()[2];
			double f1_score = 2*precision*recall/(precision + recall); 
			
//			System.out.println(String.format("%-10s %20s %20s %20s", eachEval.getKey(), "precision: "+df.format(eachEval.getValue()[0])
//					+"%", "recall: "+df.format(eachEval.getValue()[1])+"%", "f1-score: "+df.format(eachEval.getValue()[2])+"%"));
			System.out.println(String.format("%-10s %20s %20s %20s", eachEval.getKey(), "precision: "+df.format(precision*100)
					+"%", "recall: "+df.format(recall*100)+"%", "f1-score: "+df.format(f1_score*100)+"%"));				
		}
		
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println();
	}
}
