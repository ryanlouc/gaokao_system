package cn.edu.nju.nlp.main;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

public class TimeAndLocMain {
	public static Map<String, double[]>	Params;
	public static Properties			Property;
	public static DoubleArrayTrieTerm	adt;

	public void load_NERModel(String prefix) {
		Util u = new Util();
		Params = u.loadParams(prefix + "ner.model");
		Property = new Properties();
		FileReader fr;
		try {
			fr = new FileReader(prefix + "model.properties");
			Property.load(fr);
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		adt = DoubleArrayTrieTerm.getInstance(prefix + "locName.txt");
	}

	/**
	 * 输入一句话 s
	 * 
	 * @return 返回所有时间列表
	 */
	public ArrayList<String> timeList(String s) {
		TimeRecognition tr = new TimeRecognition();
		return tr.findAllTime(s);
	}

	/**
	 * 输入一句话 s，判断整句是否为时间
	 * 
	 * @return 布尔值
	 */
	public boolean isTime(String s) {
		TimeRecognition tr = new TimeRecognition();
		List<String> timeList = tr.findAllTime(s);
		for (String time : timeList) {
			if (s.equals(time)) return true;
		}
		String time = tr.OddTime_Part(s);
		if (time.equals(s)) return true;
		return false;
	}

	/**
	 * 利用词典查找一句不分词的句子中的地点
	 * 
	 * @param prefix
	 * @param str
	 * @return
	 */
	private List<String> getLocInDic(String s) {
		String temp;
		List<String> result = new ArrayList<>();
		int maxSize = adt.getMax();
		while (s.length() > 0) {
			if (s.length() < maxSize)
				temp = s;
			else
				temp = s.substring(0, maxSize);
			while (temp.length() > 1)
				if (adt.exactMatchSearch(temp) >= 0) {
					result.add(temp);
					break;
				}
				else
					temp = temp.substring(0, temp.length() - 1);
			s = s.substring(temp.length());
		}
		return result;
	}

	/**
	 * 利用序列化标注得到地名
	 * 
	 * @param s
	 * @return
	 */
	private List<String> getLocBySLB(String s) {
		int beamSize = 16, mode = 1, now = 0;
		Util u = new Util();
		Set<String> locDict = u.loadDicts("dict/locdict.txt");
		Set<String> shortLocDict = u.loadDicts("dict/locdict_short.txt");

		Beam_search b = new Beam_search();
		List<String[]> nerResult = new ArrayList<>();

		List<String[]> wordList = u.genWordPosLabel(s, mode);
		List<Term> predictTerm = b.runBeamSearch(s, beamSize, TimeAndLocMain.Params, locDict, shortLocDict, now, mode);
		for (int i = 0; i < predictTerm.size(); ++i) {
			wordList.get(i)[2] = predictTerm.get(i).label;
		}
		nerResult = u.simplifyLabel(wordList);

		for (int i = 0; i < nerResult.size(); ++i) {
			if (i > 0 && nerResult.get(i - 1)[0].equals("该")
					&& (nerResult.get(i)[0].equals("城市") || nerResult.get(i)[0].equals("地区"))) {
				nerResult.get(i - 1)[2] = "LOC";
				nerResult.get(i)[2] = "LOC";
			}
			else if (nerResult.get(i)[0].equals("该地")) nerResult.get(i)[2] = "LOC";
		}

		ArrayList<String> result = new ArrayList<>();

		int index = 0;
		while (index < nerResult.size()) {
			if (nerResult.get(index)[2].equals("LOC")) {
				StringBuilder sb = new StringBuilder();
				int i = index;
				while (i < nerResult.size() && nerResult.get(i)[2].equals("LOC")) {
					sb.append(nerResult.get(i)[0]);
					++i;
				}
				result.add(sb.toString());
				index = i;
			}
			else
				index++;
		}

		return result;
	}

	/**
	 * 输入分词结果 s，词与词之间用空格隔开，如“江苏省 南京市”
	 * 
	 * @return 返回所地点列表
	 */
	public List<String> locList(String s) {
		List<String> temp = new ArrayList<>();
		temp.addAll(getLocBySLB(s));
		StringBuilder sb = new StringBuilder();
		for (String sub : s.split(" ")) {
			sb.append(sub);
		}
		temp.addAll(getLocInDic(sb.toString()));
		// 去重
		List<String> result = new ArrayList<>();
		for (int i = 0; i < temp.size(); ++i) {
			if (result.contains(temp.get(i))) result.remove(temp.get(i));

			result.add(temp.get(i));

		}
		return result;
	}

	/**
	 * 输入分词结果 s，以及图标注产生的外部词典，词与词之间用空格隔开，如“江苏省 南京市”
	 * 
	 * @return 返回所地点列表
	 */
	public List<String> locList(String s, List<String> DicFromGraph) {
		List<String> result = locList(s);
		return result;
	}

	public static void main(String[] args) {
		TimeAndLocMain talm = new TimeAndLocMain();
		talm.load_NERModel("J://louc//NERModel//");
		List<String> timeList = talm
				.timeList("突出专家抵达湘时，—1985——2000年，20世纪50~60年代图3是7、8月份近10年7级台风海平面气压分布图（单位：百帕）。读图，夏至回答第7题。");
		for (String time : timeList) {
			System.out.println(time);
		}

		boolean isTime = talm.isTime("专家抵达湘时");
		System.out.println(isTime);

		List<String> locList = talm.locList("首尔 比 韩国 首尔 等地");
		for (String loc : locList) {
			System.out.println(loc);
		}
		// ～|~|～
		System.out.println("～".equals("~"));
	}
}
