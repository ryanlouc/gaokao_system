package cn.edu.nju.nlp.test;

import java.util.List;

import cn.edu.nju.nlp.main.TimeAndLocMain;

public class TestTime {
	public static void main(String[] args) {
		TimeAndLocMain talm = new TimeAndLocMain();
		talm.load_NERModel("J://louc//NERModel//");
		List<String> timeList = talm
				.timeList("阿萨德发老师的放假啊。当台风经过时，读某日08时地面天气图(图6)和文字信息,回答10~11题。当时，某气象小组学生探讨。当时。天气图中a→b天气的空间变化 ");
		for (String time : timeList) {
			System.out.println(time);
		}

		boolean isTime = talm.isTime("专家抵达湘时");
		System.out.println(isTime);

		List<String> locList = talm.locList("该地 比 韩国 首尔 等地");
		for (String loc : locList) {
			System.out.println(loc);
		}

	}
}
