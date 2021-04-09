package isomer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class NumberTheBranch {
    public void findBranch(int[][] rootedArray, int[] rootedToMatrix,
                           int left, int right,
                           int Cnum, Set<Integer> carbonSet,
                           int mainChainLength) {
//        Recursive export
        if(carbonSet.size() == Cnum) {
            return;
        }
        if(left > right) {
            return;
        }

//        Loop to find each small branch on the branch
        for(int i = left; i <= right; i++) {
            List<ArrayList<Integer>> branchPath = new ArrayList<>();
            branchPath.add(new ArrayList<>(1));
            while (branchPath.size() != 0) {
//                Look for possible paths on each carbon
                branchPath = eachCarbonCandidate(rootedArray, rootedToMatrix[i], carbonSet);
                NumberTheMainChain numberTheMainChain = new NumberTheMainChain();
//                Reorder paths by length
                numberTheMainChain.sortPath(branchPath);
//                Removal of Coincident Path
                numberTheMainChain.removeRepeatPath(branchPath);
//                Look for possible side backbone
                List<ArrayList<Integer>> candidateBranch = findCandidateBranch(branchPath);
//                Score the side chain
                numberBranch(candidateBranch, rootedArray, rootedToMatrix, carbonSet);
                int branchChainLength = candidateBranch.size() == 0 ? 0 : candidateBranch.get(0).size();
                int end = 0;
//                for(int k : rootedToMatrix) {
//                    System.out.print(k);
//                }
                for(int j = 0; j < rootedToMatrix.length; j++) {
                    if(rootedToMatrix[j] == -1) {
                        end = j;
                        break;
                    }
                }
                findBranch(rootedArray, rootedToMatrix, end - branchChainLength + 1 , end - 1, Cnum, carbonSet, branchChainLength);
            }
        }
    }

    //    Score the side chain
    private void numberBranch(List<ArrayList<Integer>> candidateBranch, int[][] rootedArray, int[] rootedToMatrix, Set<Integer> carbonSet) {
        if(candidateBranch.size() == 0) {
            return;
        }
        if(candidateBranch.get(0).size() <= 2) {
            numberTheBranch(candidateBranch.get(0), rootedToMatrix, carbonSet);
            return;
        }

//        Put the possible side backbone into the set
        List<HashSet<Integer>> pathSet = new ArrayList<>();
        putChainInSet(candidateBranch, pathSet);

//        score
        int[][] score = new int[candidateBranch.size()][candidateBranch.get(0).size()];
        scoreTheChain(candidateBranch, rootedArray, pathSet, score);
        int[] totalScore = new int[candidateBranch.size()];
        int minScore = Integer.MAX_VALUE;
        int minIndex = 0;
        for(int i = 0; i < candidateBranch.size(); i++) {
            for(int j = 0; j < candidateBranch.get(i).size(); j++) {
                totalScore[i] += (j + 1) * score[i][j];
            }
            if(totalScore[i] < minScore) {
                minScore = totalScore[i];
                minIndex = i;
            }
        }
//        Number the side chain
        numberTheBranch(candidateBranch.get(minIndex), rootedToMatrix, carbonSet);
    }

    static void scoreTheChain(List<ArrayList<Integer>> candidateBranch, int[][] rootedArray, List<HashSet<Integer>> pathSet, int[][] score) {
        for(int i = 0; i < candidateBranch.size(); i++) {
            if(candidateBranch.get(0).size() > 1) {
                for (int j = 1; j < candidateBranch.get(i).size() - 1; j++) {
                    List<ArrayList<Integer>> branchPath = new ArrayList<>();
                    for(int k = 0; k < rootedArray.length; k++) {
                        if(rootedArray[k][0] == candidateBranch.get(i).get(j)
                                && rootedArray[k][1] != candidateBranch.get(i).get(j - 1)
                                && rootedArray[k][1] != candidateBranch.get(i).get(j + 1)) {
                            List<Integer> temp = new ArrayList<>();
                            temp.add(rootedArray[k][0]);
                            temp.add(rootedArray[k][1]);
                            pathSet.get(i).add(rootedArray[k][1]);
                            score[i][j]++;
                            branchPath.add((ArrayList<Integer>) temp);
                        } else if(rootedArray[k][1] == candidateBranch.get(i).get(j)
                                && rootedArray[k][0] != candidateBranch.get(i).get(j - 1)
                                && rootedArray[k][0] != candidateBranch.get(i).get(j + 1)) {
                            List<Integer> temp = new ArrayList<>();
                            temp.add(rootedArray[k][1]);
                            temp.add(rootedArray[k][0]);
                            pathSet.get(i).add(rootedArray[k][0]);
                            score[i][j]++;
                            branchPath.add((ArrayList<Integer>) temp);
                        }
                    }

//                  Loop for all the atoms on the side chain
                    boolean flag = true;
                    while (flag) {
                        flag = false;
                        for(int idx = 0; idx < rootedArray.length; idx++) {
                            for(int iex = 0; iex < branchPath.size(); iex++) {
                                if((!pathSet.get(i).contains(rootedArray[idx][1]))
                                        && rootedArray[idx][0] == branchPath.get(iex).get(branchPath.get(iex).size() - 1)) {
                                    List<Integer> temp = new ArrayList<>(branchPath.get(iex));
                                    temp.add(rootedArray[idx][1]);
                                    pathSet.get(i).add(rootedArray[idx][1]);
                                    score[i][j]++;
                                    branchPath.add((ArrayList<Integer>) temp);
                                    flag = true;
                                } else if((!pathSet.get(i).contains(rootedArray[idx][0]))
                                        && rootedArray[idx][1] == branchPath.get(iex).get(branchPath.get(iex).size() - 1)) {
                                    List<Integer> temp = new ArrayList<>(branchPath.get(iex));
                                    temp.add(rootedArray[idx][0]);
                                    pathSet.get(i).add(rootedArray[idx][0]);
                                    score[i][j]++;
                                    branchPath.add((ArrayList<Integer>) temp);
                                    flag = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void putChainInSet(List<ArrayList<Integer>> candidateBranch, List<HashSet<Integer>> pathSet) {
        for(int i = 0; i < candidateBranch.size(); i++) {
            Set<Integer> tmpSet = new HashSet<>();
            for(Integer j : candidateBranch.get(i)) {
                tmpSet.add(j);
            }
            pathSet.add((HashSet<Integer>) tmpSet);
        }
    }

    //    Number the side chain
    private void numberTheBranch(List<Integer> mainBranch, int[] rootedToMatrix, Set<Integer> carbonSet) {
        int index = 0;
        while (rootedToMatrix[index] != -1) {
            index++;
        }
        for(int i = 1; i < mainBranch.size(); i++) {
            rootedToMatrix[i + index - 1] = mainBranch.get(i);
            carbonSet.add(mainBranch.get(i));
        }
    }

    //    Look for possible paths on each carbon
    public List<ArrayList<Integer>> eachCarbonCandidate(int[][] rootedArray, int currentCarbon, Set<Integer> carbonSet) {
        List<ArrayList<Integer>> branchPath = new ArrayList<>();
        Set<Integer> branchPathSet = new HashSet<>();
        for(int i = 0; i < rootedArray.length; i++) {
            if(rootedArray[i][0] == currentCarbon && (!carbonSet.contains(rootedArray[i][1]))) {
                List<Integer> temp = new ArrayList<>();
                temp.add(rootedArray[i][0]);
                temp.add(rootedArray[i][1]);
                branchPathSet.add(rootedArray[i][0]);
                branchPathSet.add(rootedArray[i][1]);
                branchPath.add((ArrayList<Integer>) temp);
            } else if(rootedArray[i][1] == currentCarbon && (!carbonSet.contains(rootedArray[i][0]))){
                List<Integer> temp = new ArrayList<>();
                temp.add(rootedArray[i][1]);
                temp.add(rootedArray[i][0]);
                branchPathSet.add(rootedArray[i][1]);
                branchPathSet.add(rootedArray[i][0]);
                branchPath.add((ArrayList<Integer>) temp);
            }
        }

//        Loop for all the atoms on the side chain
        boolean flag = true;
        while (flag) {
            flag = false;
            for(int i = 0; i < rootedArray.length; i++) {
                for(int j = 0; j < branchPath.size(); j++) {
                    if((!branchPathSet.contains(rootedArray[i][1]))
                            && rootedArray[i][0] == branchPath.get(j).get(branchPath.get(j).size() - 1)) {
                        List<Integer> temp = new ArrayList<>(branchPath.get(j));
                        temp.add(rootedArray[i][1]);
                        branchPathSet.add(rootedArray[i][1]);
                        branchPath.add((ArrayList<Integer>) temp);
                        flag = true;
                    } else if((!branchPathSet.contains(rootedArray[i][0]))
                            && rootedArray[i][1] == branchPath.get(j).get(branchPath.get(j).size() - 1)) {
                        List<Integer> temp = new ArrayList<>(branchPath.get(j));
                        temp.add(rootedArray[i][0]);
                        branchPathSet.add(rootedArray[i][0]);
                        branchPath.add((ArrayList<Integer>) temp);
                        flag = true;
                    }
                }
            }
        }
        return branchPath;
    }

    //    Look for possible side backbone
    private List<ArrayList<Integer>> findCandidateBranch(List<ArrayList<Integer>> branchPath) {
        List<ArrayList<Integer>> candidateBranch = new ArrayList<>();
        int maxLength = 0;
        for(ArrayList list : branchPath) {
            maxLength = Math.max(maxLength, list.size());
        }
        for(int i = 0; i < branchPath.size(); i++) {
            if(branchPath.get(i).size() == maxLength) {
                candidateBranch.add(new ArrayList<>(branchPath.get(i)));
            }
        }
        return candidateBranch;
    }
}
