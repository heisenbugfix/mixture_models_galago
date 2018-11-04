package edu.umass.cs.cs646;

import java.io.IOException;
import java.util.*;

import com.sun.xml.internal.ws.api.ha.StickyFeature;
import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.index.stats.NodeStatistics;
import org.lemurproject.galago.core.parse.stem.Stemmer;
import org.lemurproject.galago.core.retrieval.Results;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.prf.ExpansionModel;
import org.lemurproject.galago.core.retrieval.prf.RelevanceModel1;
import org.lemurproject.galago.core.retrieval.prf.WeightedTerm;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.util.WordLists;
import org.lemurproject.galago.utility.Parameters;

/**
 * @author Hamed Zamani (zamani@cs.umass.edu)
 */
public class MixtureFeedbackModel implements ExpansionModel {
    protected Retrieval retrieval;
    int defaultFbDocs, defaultFbTerms;
    double defaultFbOrigWeight;
    Set<String> exclusionTerms;
    Stemmer stemmer;

    public MixtureFeedbackModel(Retrieval r) throws IOException {
        retrieval = r;
        defaultFbDocs = (int) Math.round(r.getGlobalParameters().get("fbDocs", 10.0));
        defaultFbTerms = (int) Math.round(r.getGlobalParameters().get("fbTerm", 100.0));
        defaultFbOrigWeight = r.getGlobalParameters().get("fbOrigWeight", 0.2);
        exclusionTerms = WordLists.getWordList(r.getGlobalParameters().get("rmstopwords", "rmstop"));
        Parameters gblParms = r.getGlobalParameters();
        this.stemmer = FeedbackData.getStemmer(gblParms, retrieval);
    }

    public List<ScoredDocument> collectInitialResults(Node transformed, Parameters fbParams) throws Exception {
        Results results = retrieval.executeQuery(transformed, fbParams);
        List<ScoredDocument> res = results.scoredDocuments;
        if (res.isEmpty())
            throw new Exception("No feedback documents found!");
        return res;
    }

    public Node generateExpansionQuery(List<WeightedTerm> weightedTerms, int fbTerms) throws IOException, Exception {
        Node expNode = new Node("combine");
        System.err.println("Feedback Terms:");
        for (int i = 0; i < Math.min(weightedTerms.size(), fbTerms); i++) {
            Node expChild = new Node("text", weightedTerms.get(i).getTerm());
            expNode.addChild(expChild);
            expNode.getNodeParameters().set("" + i, weightedTerms.get(i).getWeight());
        }
        return expNode;
    }

    public int getFbDocCount(Node root, Parameters queryParameters) throws Exception {
        int fbDocs = (int) Math.round(root.getNodeParameters().get("fbDocs", queryParameters.get("fbDocs", (double) defaultFbDocs)));
        if (fbDocs <= 0)
            throw new Exception("Invalid number of feedback documents!");
        return fbDocs;
    }

    public int getFbTermCount(Node root, Parameters queryParameters) throws Exception {
        int fbTerms = (int) Math.round(root.getNodeParameters().get("fbTerm", queryParameters.get("fbTerm", (double) defaultFbTerms)));
        if (fbTerms <= 0)
            throw new Exception("Invalid number of feedback terms!");
        return fbTerms;
    }

    public Node interpolate(Node root, Node expandedQuery, Parameters queryParameters) throws Exception {
        queryParameters.set("defaultFbOrigWeight", defaultFbOrigWeight);
        queryParameters.set("fbOrigWeight", queryParameters.get("fbOrigWeight", defaultFbOrigWeight));
        return linearInterpolation(root, expandedQuery, queryParameters);
    }

    public Node linearInterpolation(Node root, Node expNode, Parameters parameters) throws Exception {
        double defaultFbOrigWeight = parameters.get("defaultFbOrigWeight", -1.0);
        if (defaultFbOrigWeight < 0)
            throw new Exception("There is not defaultFbOrigWeight parameter value");
        double fbOrigWeight = parameters.get("fbOrigWeight", defaultFbOrigWeight);
        if (fbOrigWeight == 1.0) {
            return root;
        }
        Node result = new Node("combine");
        result.addChild(root);
        result.addChild(expNode);
        result.getNodeParameters().set("0", fbOrigWeight);
        result.getNodeParameters().set("1", 1.0 - fbOrigWeight);
        return result;
    }

    public Parameters getFbParameters(Node root, Parameters queryParameters) throws Exception {
        Parameters fbParams = Parameters.create();
        fbParams.set("requested", getFbDocCount(root, queryParameters));
        fbParams.set("passageQuery", false);
        fbParams.set("extentQuery", false);
        fbParams.setBackoff(queryParameters);
        return fbParams;
    }

    @Override
    public Node expand(Node root, Parameters queryParameters) throws Exception {
        int fbTerms = getFbTermCount(root, queryParameters);
        // transform query to ensure it will run
        Parameters fbParams = getFbParameters(root, queryParameters);
        Node transformed = retrieval.transformQuery(root.clone(), fbParams);

        // get some initial results
        List<ScoredDocument> initialResults = collectInitialResults(transformed, fbParams);


        // extract grams from results
        Set<String> queryTerms = getTerms(stemmer, StructuredQuery.findQueryTerms(transformed));
        FeedbackData feedbackData = new FeedbackData(retrieval, exclusionTerms, initialResults, fbParams);
        List<WeightedTerm> weightedTerms = computeWeights(feedbackData, fbParams, queryParameters);
        Collections.sort(weightedTerms);
        Node expNode = generateExpansionQuery(weightedTerms, fbTerms);

        return interpolate(root, expNode, queryParameters);
    }

    public static Set<String> getTerms(Stemmer stemmer, Set<String> terms) {
        if (stemmer == null)
            return terms;

        Set<String> stems = new HashSet<String>(terms.size());
        for (String t : terms) {
            String s = stemmer.stem(t);
            stems.add(s);
        }
        return stems;
    }

    //computeWeights function returns a list of terms with their weights extracted from the feedback docs
    public List<WeightedTerm> computeWeights(FeedbackData feedbackData, Parameters fbParam, Parameters queryParameters) throws Exception {
        try {
            List<WeightedTerm> wt = new ArrayList<>();
            HashMap<String, Double> p_ThetaF = new HashMap<>();
            HashMap<String, Double> p_ThetaC = new HashMap<>();
            HashMap<String, Double> p_Zt = new HashMap<>();
            Map<String, Map<ScoredDocument, Integer>> termCounts = feedbackData.termCounts;

//            Set<String> queryTerms = feedbackData.stemmedQueryTerms;
            Set<String> queryTerms = new HashSet<>();
            for(Map.Entry<String, Map<ScoredDocument, Integer>> t: termCounts.entrySet()){
                queryTerms.add(t.getKey());
            }
            Set<String> excTerms = feedbackData.exclusionTerms;
            double lambda = 1.0 - fbParam.getDouble("fbOrigWeight");
            //get corpus length
            Retrieval r = feedbackData.retrieval;
            Node n = new Node();
            n.setOperator("lengths");
            n.getNodeParameters().set("part", "lengths");
            FieldStatistics stat = retrieval.getCollectionStatistics(n);
            double corpusLen = stat.documentCount;
            //removing exclusions
            if(excTerms!=null) queryTerms.removeAll(excTerms);
            //initialize p_ThetaF
            for(String s : queryTerms){
                p_ThetaF.put(s, 1.0/queryTerms.size());
            }
            //calculate p_ThetaC
            for(String s : queryTerms){
                double cT_F = termFreqInCorpus(s, corpusLen, retrieval);
                p_ThetaC.put(s, cT_F);
            }
            for(int i = 0 ; i < 1000 ; i++) {
                //E:Step
                for (String s : queryTerms) {
                    double ptf = p_ThetaF.get(s);
                    double num = (1.0 - lambda)*ptf;
                    double denom = (1.0 - lambda)*ptf + lambda*p_ThetaC.get(s);
                    double pzt = num/denom;
                    p_Zt.put(s, pzt);
                }
                //M:step
                double normalization = 0.0;
                for(String s : queryTerms) {
                    normalization+= termFreqInRel(s, feedbackData)*p_Zt.get(s);
                }
                for(String s : queryTerms) {
                    double num = termFreqInRel(s, feedbackData)*p_Zt.get(s);
                    p_ThetaF.put(s, num/normalization);
                }
            }
            for(Map.Entry<String, Double> m : p_ThetaF.entrySet()){
                WeightedUnigram w = new WeightedUnigram(m.getKey(), m.getValue());
                wt.add(w);
            }
            return wt;
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("This should be implemented! This method outputs a list of terms with weights.");
        }
    }

    private double termFreqInCorpus(String s, double corpusLen, Retrieval retrieval){
        try {
            String que = s;
            Node node = StructuredQuery.parse(que);
            node.getNodeParameters().set("queryType", "count");
            node = retrieval.transformQuery(node, Parameters.create());
            NodeStatistics stat2 = retrieval.getNodeStatistics(node);
            return stat2.nodeDocumentCount/corpusLen;
        }catch (Exception e){
            e.printStackTrace();
            throw new NoSuchElementException();
        }
    }

    private double termFreqInRel(String s, FeedbackData fbData){
        Map<String, Map<ScoredDocument, Integer>> termCounts = fbData.termCounts;
        Map<ScoredDocument, Integer> map = termCounts.get(s);
        double ans = 0.0;
        for(Map.Entry<ScoredDocument, Integer> m : map.entrySet()){
            ans += m.getValue();
        }
        return ans;
    }
}
