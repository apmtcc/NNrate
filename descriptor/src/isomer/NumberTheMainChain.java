package isomer;

import java.util.*;

public class NumberTheMainChain {
    public AlkaneInfo findMainChain(int[][] rootedArray, int Cnum){
        rootedArray = sortRootedArray(rootedArray);
//        Store possible paths
        List<ArrayList<Integer>> path = new ArrayList<>();
        for(int i = 0; i < rootedArray.length; i++) {
            if(rootedArray[i][0] == 0) {
                List<Integer> temp = new ArrayList<>();
                temp.add(rootedArray[i][0]);
                temp.add(rootedArray[i][1]);
                path.add((ArrayList<Integer>) temp);
            } else {
                for(int j =0; j < path.size();j++) {

                    if(rootedArray[i][0] == path.get(j).get(path.get(j).size()-1) ) {
                        List<Integer> temp = new ArrayList<>(path.get(j));
                        temp.add(rootedArray[i][1]);
                        path.add((ArrayList<Integer>) temp);
                    }
                }
            }

        }

//        Reorder paths by length
        sortPath(path);

//        Remove coincident short paths
        removeRepeatPath(path);

//        Look for possible mainchains
        List<ArrayList<Integer>> candidateMainChain = findCandidateMainChain(path, rootedArray);

//        Main chain number
        int[] rootedToMatrix = new int[rootedArray.length + 1];
//        The main chain has an initial value of 1
        for(int i = 0; i < rootedToMatrix.length; i++) {
            rootedToMatrix[i] = -1;
        }
        int mainChainLength = numberMainChain(candidateMainChain, rootedArray, rootedToMatrix);
        Set<Integer> carbonSet = new HashSet<>();
        for(int i = 0; i < mainChainLength; i++) {
            carbonSet.add(rootedToMatrix[i]);
        }
        NumberTheBranch numberTheBranch = new NumberTheBranch();
        numberTheBranch.findBranch(rootedArray, rootedToMatrix, 0, mainChainLength - 1, Cnum, carbonSet, mainChainLength);
        String str = "";
        for(int i = 0; i < rootedToMatrix.length; i++) {
            str += rootedToMatrix[i] + " ";
        }

        int[][] adjacencyMatrix = matrixToAdjacencyMatrix(rootedArray, rootedToMatrix);
        AlkaneInfo alkaneInfo = new AlkaneInfo();
        alkaneInfo.setAdjacencyMatrix(adjacencyMatrix);
        alkaneInfo.setRootedArray(rootedArray);
        return alkaneInfo;
    }

    //     distance matrix
    private int[][] matrixToAdjacencyMatrix(int[][] rootedArray, int[] rootedToMatrix) {
        int Cnum = rootedToMatrix.length;
        int[][] adjacencyMatrix = new int[Cnum][Cnum];
        for(int i = 0; i < rootedArray.length; i++) {
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < Cnum; k++) {
                    if(rootedArray[i][j] == rootedToMatrix[k]) {
                        rootedArray[i][j] = k;
                        break;
                    }
                }
            }
        }

        for(int i = 0; i < rootedArray.length; i++) {
            int x = rootedArray[i][0];
            int y = rootedArray[i][1];
            adjacencyMatrix[x][y] = 1;
            adjacencyMatrix[y][x] = 1;

        }
        return adjacencyMatrix;
    }


    //    reorder
    public int[][] sortRootedArray(int[][] rootedArray) {
        for(int i = 0; i < rootedArray.length;i++) {
            if(rootedArray[i][0] > rootedArray[i][1]) {
                int temp = rootedArray[i][0];
                rootedArray[i][0] = rootedArray[i][1];
                rootedArray[i][1] = temp;
            }
        }
        return rootedArray;
    }

    //    Path reordering
    public void sortPath(List<ArrayList<Integer>> path) {
        Collections.sort(path, (o1, o2) -> {
            if(o1.size() < o2.size()) {
                return 1;
            } else {
                return -1;
            }
        });
    }

    //    Remove coincident short paths
    public void removeRepeatPath(List<ArrayList<Integer>> path) {
//        Mark coincidence index
        boolean[] pathFlag = new boolean[path.size()];
        for(int i = 0; i < path.size(); i++) {
            pathFlag[i] = false;
        }
        for(int i = 0; i < path.size() - 1;i++) {
            for(int j = i + 1; j < path.size(); j++) {
                int k = 0;
                while (k < path.get(j).size()) {
                    if(path.get(j).get(k) == path.get(i).get(k)) {
                        k++;
                        if(k == path.get(j).size()) {
                            pathFlag[j] = true;
                        }
                    } else {
                        break;
                    }
                }

            }
        }

//        Delete according to coincidence index
        for(int i = path.size()-1; i >= 0; i--) {
            if(pathFlag[i]) {
                path.remove(i);
            }
        }
    }

    //    Look for possible mainchains
    public List<ArrayList<Integer>> findCandidateMainChain(List<ArrayList<Integer>> path, int[][] rootedArray) {

        List<ArrayList<Integer>> candidateMainChain = new ArrayList<>();

        for(int i = 0; i < path.size() - 1; i++) {
            for(int j = i + 1; j < path.size(); j++) {
                if(path.get(i).get(1) != path.get(j).get(1)) {
                    List<Integer> temp = new ArrayList<>();
                    for(int k = path.get(i).size()-1; k >= 0; k--) {
                        temp.add(path.get(i).get(k));
                    }
                    for(int k = 1; k < path.get(j).size(); k++) {
                        temp.add(path.get(j).get(k));
                    }
                    candidateMainChain.add(new ArrayList<>(temp));
                    temp.clear();
                }
            }
        }

        int maxLength = 0;
        for(ArrayList list : candidateMainChain) {
            maxLength = Math.max(maxLength, list.size());
        }
        for(int i = candidateMainChain.size() - 1; i >= 0; i--) {
            if(candidateMainChain.get(i).size() != maxLength) {
                candidateMainChain.remove(i);
            }
        }

//        Maximum number of substituents in the main chain
        List<Integer> numberOfSubstituent = new ArrayList<>();
        for(ArrayList list: candidateMainChain) {
            Set<Integer> tempSet = new HashSet();
            int num = 0;
            for(Object i : list) {
                tempSet.add((int)i);
            }
            for(int i = 0; i < rootedArray.length; i++) {
                if(tempSet.contains(rootedArray[i][0]) && (!tempSet.contains(rootedArray[i][1]))) {
                    num++;
                } else if (tempSet.contains(rootedArray[i][1]) && (!tempSet.contains(rootedArray[i][0]))) {
                    num++;
                }
            }
            numberOfSubstituent.add(num);
        }
        int maxSubstituent = Integer.MIN_VALUE;
        for(Integer i : numberOfSubstituent) {
            maxSubstituent = maxSubstituent > i ? maxSubstituent : i;
        }
        for(int i = candidateMainChain.size() - 1; i >= 0; i--) {
            if(maxSubstituent != numberOfSubstituent.get(i)) {
                candidateMainChain.remove(i);
            }
        }
        return candidateMainChain;
    }

    //    Main chain number
    public int numberMainChain(List<ArrayList<Integer>> candidateMainChain, int[][] rootedArray, int[] rootedToMatrix) {
//        Put the possible backbone into the set
        List<HashSet<Integer>> pathSet = new ArrayList<>();
        NumberTheBranch.putChainInSet(candidateMainChain, pathSet);

//        score
        int[][] score = new int[candidateMainChain.size()][candidateMainChain.get(0).size()];
        NumberTheBranch.scoreTheChain(candidateMainChain, rootedArray, pathSet, score);

        int[][] totalScore = new int[candidateMainChain.size()][2];
        int minScore = Integer.MAX_VALUE;
        int minIndex = 0;
        boolean positive = false;
        for(int i = 0; i < candidateMainChain.size(); i++) {
            for(int j = 0; j < candidateMainChain.get(i).size(); j++) {
//                Numbering from front to back
                totalScore[i][0] += (j + 1) * score[i][j];
//                Numbers from the back to the front
                totalScore[i][1] += (candidateMainChain.get(i).size() - j) * score[i][j];

            }
            if(totalScore[i][0] <= totalScore[i][1]) {
                if(totalScore[i][0] < minScore) {
                    minScore = Math.min(minScore, totalScore[i][0]);
                    minIndex = i;
                    positive = true;
                }
            } else {
                if(totalScore[i][1] < minScore) {
                    minScore = Math.min(minScore, totalScore[i][1]);
                    minIndex = i;
                    positive = false;
                }
            }
        }

//        Number the main chain
        if(positive) {
            for(int i = 0; i < candidateMainChain.get(minIndex).size(); i++) {
                rootedToMatrix[i] = candidateMainChain.get(minIndex).get(i);
            }
        } else {
            for(int i = 0; i < candidateMainChain.get(minIndex).size(); i++) {
                rootedToMatrix[candidateMainChain.get(minIndex).size() - i - 1] = candidateMainChain.get(minIndex).get(i);
            }
        }
        return candidateMainChain.get(minIndex).size();
    }
}
