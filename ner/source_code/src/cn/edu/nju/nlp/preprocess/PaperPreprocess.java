package cn.edu.nju.nlp.preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

/*
 * 预处理的试卷导出需要且只要勾选 粗粒度分词 细粒度分词 时间 地点 等四个选项
 */
public class PaperPreprocess {
	public static void main(String[] args) {
		PaperPreprocess p = new PaperPreprocess();

		String dirPath = "C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\试卷";

		File dir = new File(dirPath);
		File[] files = dir.listFiles();

		for (File file : files) {
			String halfWidthFile = "C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\半角字符试卷\\" + file.getName();
			String outputFile = "C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\trainFile\\" + file.getName();

			System.out.println("文本：" + file.getName());

			FilePreprocess.fullToHalfWidth(file.getAbsolutePath(), halfWidthFile);
			p.preprocess(halfWidthFile, outputFile);

			System.out.println();
		}

	}

	/*
	 * @param inputFile: 需要处理的文件 outputFile: 处理之后的输出文件 功能说明： 将从网站上下载的具有 粗粒度分词
	 * 细粒度分词 时间 地点 的试卷进行处理 得到与日报格式一致的试卷，这样才可以与日报放在一起进行训练
	 */
	public void preprocess(String inputFile, String outputFile) {
		ArrayList<String> senList = loadFile(inputFile);
		ArrayList<ArrayList<String[]>> wordList = new ArrayList<ArrayList<String[]>>();

		for (String sen : senList) {
			ArrayList<String[]> wordPosLabel = genWordPosLabel(sen);
			if (wordPosLabel == null) return; // 人工分词错误处理
			wordList.add(wordPosLabel);
		}

		BufferedWriter bw = null;
		try {
			// bw = new BufferedWriter(new FileWriter(outputFile));
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "utf-8"));
			for (ArrayList<String[]> wplList : wordList) {
				for (String[] wpl : wplList)
					bw.write(wpl[0] + '/' + "NULL" + '/' + wpl[2] + ' ');
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
			String line = br.readLine();// 去除试卷中第一行
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

	/*
	 * 对从原文件中读入的一行进行处理 处理方式：先对粗粒度分词进行时间地点标注 再调用genFineGrainWordPosLabel函数
	 * 对照细粒度分词 改为 进行细粒度分词标注 细粒度分词后再对标注进行修改 连续的loc 改为 b_ns i_ns... 连续的time改为 b_t
	 * i_t... 返回标注结果
	 */
	private ArrayList<String[]> genWordPosLabel(String sen) {
		ArrayList<String[]> wordPosLabel = new ArrayList<String[]>();

		String[] WordTimeLoc = (sen + " !@#").split("!@#");

		String[] words = WordTimeLoc[0].split(" ");
		for (String word : words) {
			String[] wpl = new String[3];
			wpl[0] = word.trim();
			wpl[1] = null;
			wpl[2] = "other";
			wordPosLabel.add(wpl);
		}

		if (WordTimeLoc[2] != null) {
			String[] times = WordTimeLoc[2].split(" ");
			for (String time : times) {
				if (time.trim() == null || time.trim().length() <= 0) continue;
				int num = StringToNum(time.trim());

				try {
					wordPosLabel.get(num)[2] = "time";
				}
				catch (Exception e) {
					System.out.println("数组越界，原因可能是分词标注错误-标注的词语下标：" + num + " 分词的词语实际数目：" + wordPosLabel.size());
					return null;
				}
			}
		}

		if (WordTimeLoc[3] != null) {
			String[] locs = WordTimeLoc[3].split(" ");
			for (String loc : locs) {
				if (loc.trim() == null || loc.trim().length() <= 0) continue;
				int num = StringToNum(loc.trim());

				try {
					wordPosLabel.get(num)[2] = "loc";
				}
				catch (Exception e) {
					System.out.println("数组越界，原因可能是分词标注错误-标注的词语下标：" + num + " 分词的词语实际数目：" + wordPosLabel.size());
					return null;
				}
			}
		}

		return modifyLabel(genFineGrainWordPosLabel(wordPosLabel, WordTimeLoc[1]));
	}

	/*
	 * 因为时间，地点的标注是对粗粒度分词之后以序号的方式给出 因此需要将字符串形式的序号改为整数
	 */
	private int StringToNum(String s) {
		int num = 0;
		int length = s.length();
		for (int i = 0; i < length; i++)
			num = num * 10 + (s.charAt(i) - '0');
		return num;
	}

	/*
	 * 主要在粗粒度分词标注的基础上，修改为细粒度分词标注 方法很简单，直接对照细粒度分词结果修改就行 例如之前的 巴拿马运河/LOC 细粒度分词中有
	 * 巴拿马 运河 因此对照后直接修改为 巴拿马/LOC 运河/LOC
	 */
	private ArrayList<String[]> genFineGrainWordPosLabel(ArrayList<String[]> coarseGrainWordPosLabel,
			String fineGrainSen) {
		ArrayList<String[]> fineGrainWordPosLabel = new ArrayList<String[]>();

		String[] words = fineGrainSen.trim().split(" ");

		int j = 0;
		int size = words.length;
		for (String[] cgwpl : coarseGrainWordPosLabel) {
			String word = "";

			try {

				while (!cgwpl[0].trim().equals(word)) {
					String[] fgwpl = new String[3];
					fgwpl[0] = words[j].trim();
					fgwpl[1] = null;
					fgwpl[2] = cgwpl[2];
					fineGrainWordPosLabel.add(fgwpl);

					word += words[j].trim();
					j++;
				}
			}
			catch (Exception e) {
				// 人工分词错误处理
				System.out.println("粗粒度分词单元：" + cgwpl[0]);
				System.out.println("细粒度分词拼凑结果：" + word);
				System.out.println("本行细粒度分词结果：" + fineGrainSen);
				return null;
			}
		}

		return fineGrainWordPosLabel;
	}

	/*
	 * 将标注结果进行修改 连续的地点改为 b_ns i_ns... 连续的时间改为 b_t i_t... 例如 江苏省/LOC 南京市/LOC
	 * 栖霞区/LOC 改为 江苏省/b_ns 南京市/i_ns 栖霞区/i_ns
	 */
	private ArrayList<String[]> modifyLabel(ArrayList<String[]> wordPosLabel) {
		if (wordPosLabel == null) return null; // 人工分词错误处理

		int size = wordPosLabel.size();

		for (int i = size - 1; i > 0; i--) {
			String[] cur = wordPosLabel.get(i);
			String[] pre = wordPosLabel.get(i - 1);
			if (cur[2].equals("time")) {
				if (pre[2].equals("time"))
					cur[2] = "i_t";
				else
					cur[2] = "b_t";
			}
			else if (cur[2].equals("loc")) {
				if (pre[2].equals("loc"))
					cur[2] = "i_ns";
				else
					cur[2] = "b_ns";
			}
		}

		if (wordPosLabel.get(0)[2].equals("time"))
			wordPosLabel.get(0)[2] = "b_t";
		else if (wordPosLabel.get(0)[2].equals("loc")) wordPosLabel.get(0)[2] = "b_ns";

		return wordPosLabel;
	}
}
