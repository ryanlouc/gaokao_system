package cn.edu.nju.nlp.main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Beam_search {

	/**
	 * 运行beam search
	 * 
	 * @param wordListence
	 *            原始句子
	 * @param beam_size
	 *            beam大小
	 * @param params
	 *            double数组中[0：参数总和，1：上次更新的时间，2：参数的值]
	 * @param actions
	 *            由Term的label生成 ???
	 * @param now
	 *            第i次迭代的第j个句子序号
	 * @param mode
	 *            0表示训练，1表示测试
	 * @return
	 */
	List<Term> runBeamSearch(String sentence, int beamSize, Map<String, double[]> params, Set<String> locDict,
			Set<String> shortLocDict, double now, int mode, boolean use_postag, String[] all_labels) {

		Util u = new Util();
		List<String[]> wordList = u.genWordPosLabel(sentence, mode, use_postag);

		// beam search
		ArrayList<Term> agenda = new ArrayList<Term>();
		ArrayList<Term> tempAgenda = new ArrayList<Term>();
		Term first = new Term(0.0, -1, null, null);
		tempAgenda.add(first);

		for (int i = 0; i < wordList.size(); ++i) {
			agenda.clear();
			for (Term term : tempAgenda) {
				for (String label : all_labels) {
					double score = computScore(wordList, i, new Term(0.0, i, label, term), params, locDict,
							shortLocDict, mode, use_postag);
					updateAgenda(agenda, new Term(term.score + score, i, label, term), beamSize);
				}

			}
			tempAgenda.clear();
			tempAgenda.addAll(agenda);
		}

		List<Term> predictTerm = new ArrayList<>();
		if (mode == 0) {
			List<Term> correctTerm = u.genTerm(wordList);
			String[] correctLabels = extractLabels(correctTerm.get(correctTerm.size() - 1));

			Term best = agenda.get(agenda.size() - 1);
			predictTerm = best.backTrace(best);
			assert (predictTerm.size() == correctTerm.size());
			String[] predictLabels = extractLabels(agenda.get(agenda.size() - 1));

			for (int i = 0; i < wordList.size(); ++i) {
				List<String> correctFeatures = extractFeature(wordList, correctLabels, i, locDict, shortLocDict, use_postag);
				List<String> predictFeatures = extractFeature(wordList, predictLabels, i, locDict, shortLocDict, use_postag);
				double correctScale = 1.0;
				double predictScale = -1.0;

				lazy_update(params, correctFeatures, now, correctScale);
				lazy_update(params, predictFeatures, now, predictScale);
			}
		}
		else {
			Term best = agenda.get(agenda.size() - 1);
			predictTerm = best.backTrace(best);
		}

		return predictTerm;
	}

	/**
	 * 根据term的标签生成action，从后往前生成
	 * 
	 * @param term
	 *            大小为sen.size
	 */
	String[] extractLabels(Term t) {
		List<Term> term = t.backTrace(t);
		String[] labels = new String[term.size()];
		for (int i = 0; i < term.size(); ++i) {
			labels[i] = term.get(i).label;
		}
		return labels;
	}

	/**
	 * 更新beam search中的项，保证其中个数最大为beam_size
	 * 
	 * @param agenda
	 * @param term
	 * @param beam_size
	 */
	void updateAgenda(List<Term> agenda, Term term, int beam_size) {
		Comparator<Term> cmp = new Comparator<Term>() {
			public int compare(Term t1, Term t2) {
				if (t1.score >= t2.score)
					return 1;
				else
					return -1;
			}
		};

		if (agenda.size() < beam_size) {
			agenda.add(term);
			Collections.sort(agenda, cmp);
		}
		else if (term.score > agenda.get(0).score) {
			agenda.set(0, term);
			Collections.sort(agenda, cmp);
		}
	}

	/**
	 * 计算生成的句子片段得分
	 * 
	 * @param wordList
	 * @param i 
	 * 			    生成sent中到i为止句子的标签
	 * @param term
	 * @param params
	 *            double数组中[0：参数总和，1：上次更新的时间，2：参数的值]
	 * @param shortLocDict
	 * @param locDict
	 * @param mode
	 *            0表示训练，1表示测试
	 * @return
	 */
	double computScore(List<String[]> wordList, int i, Term term, Map<String, double[]> params, Set<String> locDict,
			Set<String> shortLocDict, int mode, boolean use_postag) {
		double result = 0.0;

		if (term.isEmpty()) {
			return result;
		}

		String[] labels = extractLabels(term);
		List<String> features = extractFeature(wordList, labels, i, locDict, shortLocDict, use_postag);

		for (String feature : features) {
			if (params.containsKey(feature)) {
				if (mode == 0)
					result += params.get(feature)[2];
				else
					result += params.get(feature)[0];
			}
		}
		return result;
	}

	void lazy_update(Map<String, double[]> params, List<String> features, double now, double scale) {
		for (String feature : features) {
			if (!params.containsKey(feature)) {
				double[] temp = { scale, now, scale };
				params.put(feature, temp);
			}
			else {
				double elapsed = now - params.get(feature)[1];
				double currVal = params.get(feature)[2];
				double currSum = params.get(feature)[0];

				params.get(feature)[0] = currSum + elapsed * currVal + scale;
				params.get(feature)[1] = now;
				params.get(feature)[2] = currVal + scale;
			}
		}
	}

	/** 判断字符的类型，分为数字(D)、字母(L)、汉字(H)和其他符号(O) */
	String charType(String word) {
		String chr = "";

		for (int i = 0; i < word.length(); ++i) {
			char ch = word.charAt(i);

			if (ch == '\u96f6' || ch == '\u4e00' || ch == '\u4e8c' || ch == '\u4e09' || ch == '\u56db'
					|| ch == '\u4e94' || ch == '\u516d' || ch == '\u4e03' || ch == '\u516b' || ch == '\u4e5d'
					|| ch == '\u5341' || ch == '\u767e' || ch == '\u5343' || ch == '\u4e07' || ch == '\u4ebf')
				chr += 'D';
			else if (ch >= '\u0030' && ch <= '\u0039')
				chr += 'd';
			else if ((ch >= '\u0041' && ch <= '\u005a') || (ch >= '\u0061' && ch <= '\u007a'))
				chr += 'l';
			else if (ch >= '\u4e00' && ch <= '\u9fa5')
				chr += 'h';
			else
				chr += 'o';
		}
		return chr;
	}

	/**
	 * 对某个词生成相应的特征(词性标注已加)
	 * 
	 * @param shortLocDict
	 * @param locDict
	 * @return
	 */
	List<String> extractFeature(List<String[]> wordList, String[] labels, int i, Set<String> locDict,
			Set<String> shortLocDict, boolean use_postag) {
		List<String> features = new ArrayList<>();

		String prev_word, curr_word, next_word;
		curr_word = wordList.get(i)[0];
		if (i == 0) {
			prev_word = "&&&&";
		}
		else {
			prev_word = wordList.get(i - 1)[0];
		}
		if (i == wordList.size() - 1) {
			next_word = "$$$$";
		}
		else {
			next_word = wordList.get(i + 1)[0];
		}
		String prev_cht = charType(prev_word);
		String curr_cht = charType(curr_word);
		String next_cht = charType(next_word);

		features.add("1=word[-1]=" + prev_word + "Label=" + labels[i]);
		features.add("2=word[0]=" + curr_word + "Label=" + labels[i]);
		features.add("3=word[+1]=" + next_word + "Label=" + labels[i]);
		features.add("4=word[-1]word[0]=" + prev_word + curr_word + "Label=" + labels[i]);
		features.add("5=word[0]word[+1]=" + curr_word + next_word + "Label=" + labels[i]);
		features.add("6=word[-1]word[0]word[+1]=" + prev_word + curr_word + next_word + "Label=" + labels[i]);
		
		features.add("7=cht[0]=" + curr_cht + "Label=" + labels[i]);
		if(!prev_word.equals("&&&&")){
			features.add("8=cht[-1]=" + prev_cht + "Label=" + labels[i]);
			features.add("9=cht[-1]cht[0]=" + prev_cht + curr_cht + "Label=" + labels[i]);
		}
		if(!next_word.equals("$$$$")){
			features.add("10=cht[+1]=" + next_cht + "Label=" + labels[i]);
			features.add("11=cht[0]cht[+1]=" + curr_cht + next_cht + "Label=" + labels[i]);
		}
		if(!prev_word.equals("&&&&") && !next_word.equals("$$$$")){
			features.add("12=cht[-1]cht[0]cht[+1]=" + prev_cht + curr_cht + next_cht + "Label=" + labels[i]);			
		}

		if (curr_word.endsWith("省") || curr_word.endsWith("市") || curr_word.endsWith("县") || curr_word.endsWith("区")
				|| curr_word.endsWith("乡") || curr_word.endsWith("镇") || curr_word.endsWith("村")
				|| curr_word.endsWith("旗") || curr_word.endsWith("州") || curr_word.endsWith("府")
				|| curr_word.endsWith("都") || curr_word.endsWith("江") || curr_word.endsWith("河")
				|| curr_word.endsWith("山") || curr_word.endsWith("湖") || curr_word.endsWith("洋")
				|| curr_word.endsWith("海") || curr_word.endsWith("岛") || curr_word.endsWith("峰")
				|| curr_word.endsWith("站") || curr_word.endsWith("街") || curr_word.endsWith("路")
				|| curr_word.endsWith("道") || curr_word.endsWith("巷") || curr_word.endsWith("里")
				|| curr_word.endsWith("町") || curr_word.endsWith("庄") || curr_word.endsWith("弄")
				|| curr_word.endsWith("堡")) {
			features.add("13=word[0]is_loc_end=" + curr_word.charAt(curr_word.length()-1) + "Label=" + labels[i]);;
		}

		if (curr_word.equals("专区") || curr_word.equals("地区") || curr_word.equals("自治区")
				|| curr_word.equals("特区") || curr_word.equals("行政区") || curr_word.equals("自治区")
				|| curr_word.equals("自治州") || curr_word.equals("自治县") || curr_word.equals("自贸区")
				|| curr_word.equals("海峡") || curr_word.equals("平原") || curr_word.equals("高原")
				|| curr_word.equals("盆地") || curr_word.equals("丘陵") || curr_word.equals("群岛")
				|| curr_word.equals("峡谷") || curr_word.equals("山脉") || curr_word.equals("大峡谷")
				|| curr_word.equals("运河")) {
			features.add("14=word[0]=equal_loc_end=" + curr_word + "Label=" + labels[i]);
		}

		if (locDict.contains(curr_word)) {
			features.add("15=in_locDict=Label=" + labels[i]);
		}

		if (shortLocDict.contains(curr_word)) {
			features.add("16=word[0]in_shortLocDict=Label=" + labels[i]);
		}
		
		if(use_postag){
			String curr_postag = wordList.get(i)[1];
			String prev_postag = null, next_postag = null;
			features.add("17=postag[0]="+curr_postag+"=Label="+labels[i]);
			if(i > 0){
				prev_postag = wordList.get(i-1)[1];
				features.add("18=postag[-1]="+prev_postag+"=Label="+labels[i]);
				features.add("19=postag[-1]postag[0]="+prev_postag+curr_postag+"=Label="+labels[i]);
			}
			if(i < wordList.size()-1){
				next_postag = wordList.get(i+1)[1];
				features.add("20=postag[+1]="+next_postag+"=Label="+labels[i]);
				features.add("21=postag[0]postag[+1]="+curr_postag+next_postag+"=Label="+labels[i]);
			}
			if(i > 0 && i < wordList.size()-1){
				features.add("22=postag[-1]postag[0]postag[+1]="+prev_postag+curr_postag+next_postag+"=Label="+labels[i]);
			}
		}
		
		return features;
	}

}
