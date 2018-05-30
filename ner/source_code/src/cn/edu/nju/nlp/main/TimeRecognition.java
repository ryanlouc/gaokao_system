package cn.edu.nju.nlp.main;

import java.util.*;
import java.util.regex.*;

public class TimeRecognition {
	public static int				YEAR			= 0;
	public static int				MONTH			= 1;
	public static int				DAY				= 2;
	public static int				HOUR			= 3;
	public static int				FULL_MONTH		= 4;
	public static int				REAL_TIME_ONE	= 5;
	public static int				REAL_TIME_TWO	= 6;
	public static int				REAL_TIME_THREE	= 7;
	public static int				FIGURE_ONE		= 8;
	public static int				FIGURE_TWO		= 9;

	public static final String[]	REGEX			= { "YEAR", "MONTH", "DAY", "HOUR", "FULL_MONTH", "REAL_TIME_ONE",
			"REAL_TIME_TWO", "REAL_TIME_THREE", "FIGURE_ONE", "FIGURE_TWO" };

	public boolean containNumber(String s) {
		for (int i = 0; i < s.length(); ++i) {
			if (Character.isDigit(s.charAt(i))) return true;
		}
		return false;
	}

	/**
	 * 识别“……期间”，“……时”，“当日” 这是整句话情况，考虑逗号
	 * 
	 * @return
	 */
	public String OddTime_Full(String s) {
		int index = s.indexOf("，");
		String time = "";
		if (index > 0 && s.charAt(index - 1) == '时' && (containNumber(s.substring(0, index)) == false)) {
			time = s.substring(0, index);
		}
		else if (index > 1 && s.charAt(index - 2) == '期' && s.charAt(index - 2) == '间') {
			time = s.substring(0, index);
		}
		else if (index > 1 && s.charAt(index - 2) == '当' && s.charAt(index - 2) == '日') {
			time = s.substring(0, index);
		}
		return time;
	}

	// 主要为main函数中：输入一句话的一部分，判断是否为时间服务
	public String OddTime_Part(String s) {
		String time = "";
		if ((s.endsWith("时") || s.endsWith("期间") || s.endsWith("当日")) && (containNumber(s) == false)) {
			time = s;
		}
		return time;
	}

	// 矫正MatchRegex匹配出来的时间
	public String CorrectTime(String s) {
		if ("".equals(s) || s.matches("[0-9]+")) {
			return "";
		}
		else if (s.matches(TimeAndLocMain.Property.getProperty(REGEX[REAL_TIME_ONE]))) {
			if (s.charAt(0) > '2' || s.charAt(3) > '5' || s.charAt(6) > '5') return "";
		}
		else if (s.matches(TimeAndLocMain.Property.getProperty(REGEX[REAL_TIME_TWO]))) {
			if (s.charAt(0) > '2' || s.charAt(3) > '5') return "";
		}
		else if (s.matches(TimeAndLocMain.Property.getProperty(REGEX[REAL_TIME_THREE]))) {
			return "";
		}

		return s;
	}

	public String MatchRegex(String s, String regex) {
		List<String> matchRegex = null;

		Pattern p = Pattern.compile(TimeAndLocMain.Property.getProperty(regex));

		Matcher matcher = p.matcher(s);

		if (matcher.find() && matcher.groupCount() >= 1) {
			matchRegex = new ArrayList<>();
			for (int i = 1; i <= matcher.groupCount(); i++) {
				String temp = matcher.group(i);
				matchRegex.add(temp);
			}
		}
		else {
			matchRegex = Collections.emptyList();
		}

		if (matchRegex.size() > 0) return (matchRegex.get(0)).trim();

		return "";
	}

	// 删除时间词之前的图1、题4之类标签
	public String deleteFigure(String s) {
		String sp = new String(s);
		String sub = "";

		sub = sp.replaceFirst(TimeAndLocMain.Property.getProperty(REGEX[FIGURE_ONE]), "");
		sub = sub.replaceAll(TimeAndLocMain.Property.getProperty(REGEX[FIGURE_TWO]), "");

		return sub;
	}

	public String realTime(String s) {

		List<String> time = new ArrayList<>();
		for (int i = TimeRecognition.YEAR; i <= FULL_MONTH; ++i) {
			time.add(CorrectTime(MatchRegex(deleteFigure(s), REGEX[i])));
		}
		time.add(OddTime_Full(s));

		int maxLen = time.get(0).length();
		String maxStr = time.get(0);

		for (String a : time) {
			if (a.length() > maxLen) {
				maxLen = a.length();
				maxStr = a;
			}
		}

		int index = maxStr.length();
		for (int i = maxStr.length() - 1; i >= 0; --i) {
			if (!Character.isDigit(maxStr.charAt(i)) && maxStr.charAt(i) != '－') {
				index = i + 1;
				break;
			}
		}
		return maxStr.substring(0, index);
	}

	public ArrayList<String> findAllTime(String s) {
		ArrayList<String> timeList = new ArrayList<>();
		while (!"".equals(realTime(s))) {
			timeList.add(realTime(s));
			s = s.replaceFirst(realTime(s), "");
		}
		String[] time = { "春季", "夏季", "秋季", "冬季", "早晨", "中午", "上午", "下午", "傍晚", "晚上", "夜间", "立春", "雨水", "惊蛰", "春分",
				"清明", "谷雨", "立夏", "小满", "芒种", "夏至", "小暑", "大暑", "立秋", "处暑", "白露", "秋分", "寒露", "霜降", "立冬", "小雪", "大雪",
				"冬至", "小寒", "大寒" };

		for (int i = 0; i < time.length; ++i) {
			for (int j = 0; j < time.length; ++j) {
				if (s.contains("从" + time[i] + "到" + time[j])) {
					timeList.add("从" + time[i] + "到" + time[j]);
					s = s.replace("从" + time[i] + "到" + time[j], "");
				}
				if (s.contains(time[i] + "到" + time[j])) {
					timeList.add(time[i] + "到" + time[j]);
					s = s.replace(time[i] + "到" + time[j], "");
				}
			}
		}
		for (int i = 0; i < time.length; ++i) {
			if (s.contains(time[i])) {
				String temp = time[i];
				if (s.indexOf(time[i]) + 2 < s.length() && s.charAt(s.indexOf(time[i]) + 2) == '日') temp += "日";
				timeList.add(temp);
			}
		}
		return timeList;
	}

	public static void main(String[] args) {
		TimeRecognition t = new TimeRecognition();
		// List<String> timeList =
		// t.findAllTime("十一月－18℃等温线向北突出专家抵达湘时，—1985——2000年，20世纪50~60年代图3是7、8月份近10年7级台风海平面气压分布图（单位：百帕）。读图，夏至回答第7题。");
		List<String> timeList = t
				.findAllTime("突出专家抵达湘时，—1985——2000年，20世纪50~60年代图3是7、8月份近10年7级台风海平面气压分布图（单位：百帕）。读图，夏至回答第7题。");

		for (int i = 0; i < timeList.size(); ++i) {
			System.out.println(timeList.get(i));
		}

	}

}
