package cn.edu.nju.nlp.preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

/*
 * 主要提供文件处理的公用方法(对试卷，日报均适用)
 */
public class FilePreprocess {

	public static void main(String[] args) {
		fullToHalfWidth("C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\paperTest.data",
				"C:\\Users\\lenovo\\Desktop\\作业\\毕业设计\\资料\\代码\\NER\\testFile\\paperTest.data");
	}

	/*
	 * 装载文件，以行的形式返回
	 */
	public static ArrayList<String> loadFile(String filename) {
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

	/*
	 * 将文件从全角转换为半角
	 * 
	 * @param inputFile: 需要处理的文件 outputFile: 处理之后的文件 inputFile outputFile可为同一个
	 */
	public static void fullToHalfWidth(String inputFile, String outputFile) {
		ArrayList<String> senList = loadFile(inputFile);

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "utf-8"));

			for (String sen : senList) {
				char[] charArray = sen.toCharArray();
				for (int i = 0; i < charArray.length; i++) {
					int charIntValue = (int) charArray[i];
					// 编码在65281与65374之间的为全角字符，转换偏移量为65248
					if (charIntValue >= 65281 && charIntValue <= 65374) {
						charArray[i] = (char) (charIntValue - 65248);
					}
				}

				bw.write(new String(charArray) + '\n');
			}

			bw.close();
		}
		catch (Exception e) {
			System.out.println(e);
		}
	}
}
