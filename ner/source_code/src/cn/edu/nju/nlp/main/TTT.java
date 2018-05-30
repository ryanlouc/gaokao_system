package cn.edu.nju.nlp.main;

import java.io.*;
import java.util.*;

public class TTT {
	public static void main(String[] args) throws Exception {
		Util u = new Util();
		List<String> sentences = u.loadFile("data/train.txt");
		List<String[]> temp = u.genWordPosLabel(sentences.get(0), 0, true);
		for(int i = 0; i < temp.size(); ++i){
			System.out.print(temp.get(i)[0]+"/"+temp.get(i)[1]+"/"+temp.get(i)[2]+" ");
		}
		
		//		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("data/tempfile.txt")));
//		for(String sentence : sentences){
//			List<String[]> wordList = new ArrayList<>();
//			String[] wordLabels = sentence.split(" ");
//			for (String wordLabel : wordLabels) {
//				String[] temp = wordLabel.split("/");
//				wordList.add(temp);
//			}
//			transformLabel(wordList);
//			for(int i = 0; i < wordList.size(); ++i){
//				bw.write(wordList.get(i)[0]+"/"+wordList.get(i)[1]+"/"+wordList.get(i)[2]+" ");
//			}
//			bw.write("\n");
//		}
//		bw.close();
	}
	
	public static void transformLabel(List<String[]> wordList){
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
}
