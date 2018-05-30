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
import java.util.List;

public class Preprocess {
	// 加载文件
	// 一行行读入filename文件中的内容到data数组中
	List<String> loadFile(String filename) {
		File file = new File(filename);
		List<String> data = new ArrayList<>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String line = br.readLine();
			if (line.startsWith("\uFEFF"))      // \uFEFF表示空格
			{
				line = line.substring(1);
			}
			while (line != null) {
				data.add(line);
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

		return data;
	}

	/**
	 * 对单独一句话生成切分好的训练数据
	 * 
	 * @param sen
	 * @return 数组中依次是lexicon,pos,label
	 */
	List<String[]> genWordPosLabel(String sen) {
		List<String[]> wordPosLabel = new ArrayList<>();
		String[] word = sen.split("  ");   // 以空格分割得到多个字符串
		for (int i = 0; i < word.length; ++i) {
			// 对于sen为这种形式时:[中国/ns 政府/n]nt，返回结果<中国,ns,nt>, <政府,n,nt>
			if (word[i].startsWith("[")) {
				List<String> temp = new ArrayList<>();
				word[i] = word[i].replace("[", "");
				temp.add(word[i]);
				while (!word[i].contains("]")) {
					i += 1;
					temp.add(word[i]);
				}
				String w = temp.get(temp.size() - 1);
				String label = w.substring(w.indexOf("]") + 1, w.length());
				w = w.substring(0, w.indexOf("]"));
				temp.set(temp.size() - 1, w);
				for (int j = 0; j < temp.size(); ++j) {
					String[] sp = temp.get(j).split("/");
					String lexicon = sp[0];
					String pos = sp[1];
					String[] t = { lexicon, pos, label };
					wordPosLabel.add(t);
				}
			}
			// 对于其它形式时：继续/v ，返回结果<继续, v, v>
			else {
				String[] sp = word[i].split("/");
				String lexicon = sp[0];
				String pos = sp[1];
				String[] t = { lexicon, pos, pos };
				wordPosLabel.add(t);
			}
		}

		wordPosLabel = modifyLabel(wordPosLabel);
		return wordPosLabel;
	}

	// 将之前处理得到的三元组里的第二个元素名字作出相应修改
	// 将需要的tag换成相应标记，\n表示名词
	List<String[]> modifyLabel(List<String[]> wordPosLabel) {
		for (int i = wordPosLabel.size() - 1; i >= 0; --i) {
			if (wordPosLabel.get(i)[2].equals("nr")) {
				wordPosLabel.get(i)[2] = "per";
			}
			else if (wordPosLabel.get(i)[2].equals("ns")) {
				wordPosLabel.get(i)[2] = "loc";
			}
			else if (wordPosLabel.get(i)[2].equals("nt")) {
				wordPosLabel.get(i)[2] = "org";
			}
			else if (wordPosLabel.get(i)[2].equals("t")) {
				wordPosLabel.get(i)[2] = "time";
			}
			else if (wordPosLabel.get(i)[2].equals("m")) {
				wordPosLabel.get(i)[2] = "num";
			}
			else {
				wordPosLabel.get(i)[2] = "other";
			}
		}
		return wordPosLabel;
	}

	// 生成训练的文件
	void genTrain(String filename, String output) {
		List<String> corpus = loadFile(filename);
		List<List<String[]>> gold = new ArrayList<>();
		for (int i = 0; i < corpus.size(); ++i) {
			List<String[]> wordPosLabel = genWordPosLabel(corpus.get(i));
			gold.add(wordPosLabel);
		}

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(output)), "utf-8"));
			for (int i = 0; i < gold.size(); ++i) {
				if (gold.get(i).size() > 12) {

					for (int j = 1; j < gold.get(i).size(); ++j) {
						if (j == gold.get(i).size() - 1)
							bw.append(gold.get(i).get(j)[0] + "/" + gold.get(i).get(j)[1] + "/" + gold.get(i).get(j)[2]);
						else
							bw.append(gold.get(i).get(j)[0] + "/" + gold.get(i).get(j)[1] + "/" + gold.get(i).get(j)[2]
									+ " ");
					}
					bw.append("\n");
				}
			}
		}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		finally {
			try {
				bw.flush();
				bw.close();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	// //简化标签
	// List<String[]> simplifyLabel(List<String[]> wordPosLabel)
	// {
	// for(int i = wordPosLabel.size() - 1 ; i >= 0 ; --i)
	// {
	// if(wordPosLabel.get(i)[2].equals("other"))
	// {
	// wordPosLabel.get(i)[2] = "OTHER";
	// }
	// if(wordPosLabel.get(i)[2].equals("b_nr") ||
	// wordPosLabel.get(i)[2].equals("i_nr"))
	// {
	// wordPosLabel.get(i)[2] = "PER";
	// }
	// if(wordPosLabel.get(i)[2].equals("b_ns") ||
	// wordPosLabel.get(i)[2].equals("i_ns"))
	// {
	// wordPosLabel.get(i)[2] = "LOC";
	// }
	// if(wordPosLabel.get(i)[2].equals("b_nt") ||
	// wordPosLabel.get(i)[2].equals("i_nt"))
	// {
	// wordPosLabel.get(i)[2] = "ORG";
	// }
	// if(wordPosLabel.get(i)[2].equals("b_t") ||
	// wordPosLabel.get(i)[2].equals("i_t"))
	// {
	// wordPosLabel.get(i)[2] = "TIME";
	// }
	// }
	// return wordPosLabel;
	// }

	// /**
	// * Term长度为sen长度+1
	// * @param sen
	// * @return
	// */
	// List<Term> genTerm(String sen)
	// {
	// List<String[]> wordPosPair = genWordPosPair(sen);
	// List<Term> terms = new ArrayList<>();
	// Term first = new Term(0.0, -1, null, null);
	// terms.add(first);
	// for(int i = 0; i < wordPosPair.size(); ++i)
	// {
	// Term temp = new Term(0.0, i, wordPosPair.get(i)[2],terms.get(terms.size()
	// - 1));
	// terms.add(temp);
	// }
	// genLabel(terms.get(terms.size() - 1));
	// return terms;
	// }
	//
	// List<String> genSent(String sen)
	// {
	// List<String[]> wordPosPair = genWordPosPair(sen);
	// List<String> sentence = new ArrayList<>();
	// for(int i = 0; i < wordPosPair.size(); ++i)
	// {
	// sentence.add(wordPosPair.get(i)[0]);
	// }
	// return sentence;
	// }
	//
	// /**
	// * 只在训练时使用
	// * 对一句话根据词性生成标签，从最后一个词往前生成
	// * @param Term t是最后一个词
	// */
	// void genLabel(Term term)
	// {
	// String label = null;
	// while(term.index != -1)
	// {
	// if(term.label.equals("nr") || term.label.equals("ns") ||
	// term.label.equals("nt") || term.label.equals("t"))
	// {
	// if(term.label.equals(term.pre.label))
	// {
	// term.label = "i_"+term.label;
	// }
	// else
	// {
	// term.label = "b_"+term.label;
	// }
	// }
	// else
	// {
	// term.label = "other";
	// }
	// term = term.pre;
	// }
	// }

}
