package cn.edu.nju.nlp.preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

public class JournalPreprocess {

	public static void main(String[] args) {
		JournalPreprocess p = new JournalPreprocess();

		FilePreprocess.fullToHalfWidth("C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\data\\train.txt",
				"C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\trainFile\\train.txt");

		p.preprocess("C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\trainFile\\train.txt",
				"C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\trainFile\\train.txt");

	}

	/*
	 * 功能说明：对日报进行处理，由于需要识别的只有时间和地点 所以将其它的标注例如人名和机构等均改为other
	 */
	public void preprocess(String inputFile, String outputFile) {
		ArrayList<String> senList = loadFile(inputFile);
		ArrayList<ArrayList<String[]>> wordList = new ArrayList<ArrayList<String[]>>();

		for (String sen : senList) {
			ArrayList<String[]> wordPosLabel = genWordPosLabel(sen);
			wordList.add(wordPosLabel);
		}

		BufferedWriter bw = null;
		try {
			// bw = new BufferedWriter(new FileWriter(outputFile));
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "utf-8"));
			for (ArrayList<String[]> wplList : wordList) {
				for (String[] wpl : wplList)
					bw.write(wpl[0] + '/' + wpl[1] + '/' + wpl[2] + ' ');
				bw.write('\n');
			}
			bw.close();
		}
		catch (Exception e) {
			System.out.println(e);
		}

	}

	private ArrayList<String> loadFile(String filename) {
		File file = new File(filename);
		ArrayList<String> data = new ArrayList<>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String line = null;
			while ((line = br.readLine()) != null)
				data.add(line.trim());

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

		return data;
	}

	private ArrayList<String[]> simplifyLabel(ArrayList<String[]> wordPosLabel) {
		for (String[] wpl : wordPosLabel) {
			if (!wpl[2].equals("b_t") && !wpl[2].equals("i_t") && !wpl[2].equals("b_ns") && !wpl[2].equals("i_ns"))
				wpl[2] = "other";
		}

		return wordPosLabel;
	}

	/*
	 * 功能说明：对日报进行处理，由于需要识别的只有时间和地点 所以将其它的标注例如人名和机构等均改为other
	 */
	private ArrayList<String[]> genWordPosLabel(String sen) {
		ArrayList<String[]> wordPosLabel = new ArrayList<String[]>();

		String[] words = sen.split(" ");
		for (String word : words) {
			if (word == null || word.length() == 0) continue;
			String[] wpl = word.split("/");
			for (int i = 0; i < wpl.length; i++)
				wpl[i] = wpl[i].trim();
			wordPosLabel.add(wpl);
		}

		return simplifyLabel(wordPosLabel);
	}
}
