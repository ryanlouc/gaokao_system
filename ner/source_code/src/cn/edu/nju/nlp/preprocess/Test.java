package cn.edu.nju.nlp.preprocess;

import java.io.BufferedReader;
import java.io.File;

import cn.edu.nju.nlp.main.TimeAndLocMain;

public class Test {
	public static void main(String[] args) {
		// String[] filename = { "corpus/199801.txt", "corpus/199802.txt",
		// "corpus/199803.txt", "corpus/199804.txt",
		// "corpus/199805.txt", "corpus/199806.txt"
		// };
		//
		// Preprocess p = new Preprocess();
		//
		// // p.genTrain(filename[0], "199801_train.txt");
		//
		// for(String file : filename)
		// {
		// p.genTrain(file, "trainFile/" + file.split("/")[1]);
		// }
		//
		// // String[] trainFilename = {
		// // "gold/199801_train.txt", "gold/199802_train.txt",
		// // "gold/199803_train.txt", "gold/199804_train.txt",
		// // "gold/199805_train.txt", "gold/199806_train.txt"
		// // };

		File dir = new File("data/examination-trainFiles");
		File[] files = dir.listFiles();
		for (File file : files) {
			if (file.getName().equals(".DS_Store")) System.out.println(file.getName());
		}

	}
}
