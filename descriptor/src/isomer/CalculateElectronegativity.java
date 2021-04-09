package isomer;

import java.util.*;

public class CalculateElectronegativity {

    //    The electronegativity of the storage group
    private HashMap groupElectronegativityMap;

    public double[] calculateGroupElectronegativity(int[][] rootedArray, int[][] adjacencyMatrix, List<List<Integer>> groupLists) {
        int Cnum = adjacencyMatrix.length;

//        Look for all the possible groups
        List<List<Integer>> allGroupList = findAllGroup(rootedArray, groupLists, Cnum);
        this.groupElectronegativityMap = new HashMap();

//        One carbon atom
        if(Cnum == 1) {
            return new double[]{Math.sqrt(2.27)};
        }

//        Look for only one
        for(int i = 0; i < Cnum; i++) {
            int numberOfKeys = 0;
            for(int num : adjacencyMatrix[i]) {
                if(num == 1) {
                    numberOfKeys++;
                }
            }
            if(numberOfKeys == 1) {
//                The electronegativity calculation results are added to the MAP
                groupElectronegativityMap.put(Integer.toString(i) + ",", groupMapInit());
            }
        }

        for(List list : allGroupList) {
//            Add list to set
            Set<Integer> listSet = new HashSet<>();
            for(Object num : list) {
                listSet.add((int) num);
            }
//            Look for root in every possible situation
            int numberOfRoot = findGroupRoot(list, adjacencyMatrix);
            listSet.remove(numberOfRoot);
//            Find Root's descendants
            int[] childRoot = {-1,-1,-1};
            childRoot = findRootChild(childRoot, numberOfRoot, list, adjacencyMatrix);
//            Find the full tree for each child node of root
            String[] childTree = new String[3];
            for(int i = 0; i < childRoot.length; i++) {
                if(childRoot[i] != -1) {
                    childTree[i] = findAllChildPath(childRoot[i], listSet, rootedArray);
                }
            }
//            Calculate the electronegativity of the groups
            double groupElectronegativity = groupEquation(childTree);
//            Join the map
            Collections.sort(list);
            String groupstr = "";
            for(Object num : list) {
                groupstr += num + ",";
            }
            groupElectronegativityMap.put(groupstr, groupElectronegativity);
        }

//        Calculate the electronegativity on each carbon
        List<Integer> carbonlist = new ArrayList<>();
        for(int i = 0; i < Cnum; i++) {
            carbonlist.add(i);
        }

//        Stores the electronegativity array
        double[] electronegativity = new double[Cnum];
        for(int i = 0; i < Cnum; i++) {
            int numberOfRoot = carbonlist.get(0);
            carbonlist.remove(0);
//            Add to the set
            Set carbonlistSet = new HashSet();
            for(int num : carbonlist) {
                carbonlistSet.add(num);
            }
            int[] childRoot = {-1, -1, -1, -1};
            childRoot = findRootChild(childRoot, numberOfRoot, carbonlist, adjacencyMatrix);

//            Find root for each full tree
            String[] childTree = new String[4];
            for(int j = 0; j < childTree.length; j++) {
                if(childRoot[j] != -1) {
                    childTree[j] = findAllChildPath(childRoot[j], carbonlistSet, rootedArray);
                }
            }
//            Calculate the electronegativity on each carbon
            electronegativity[i] = Math.sqrt(groupEquation(childTree));
            carbonlist.add(i);
        }
        return electronegativity;
    }

    private double groupEquation(String[] childTree) {
        double groupElectronegativity = GroupElectronegativity.Electronegativity.CARBON.getValue();
        for(int i = 0; i < childTree.length; i++) {
            if(groupElectronegativityMap.containsKey(childTree[i])) {
                groupElectronegativity += (double)groupElectronegativityMap.get(childTree[i]);
            } else if(childTree[i] == null){
                groupElectronegativity += GroupElectronegativity.Electronegativity.HYDROGEN.getValue();
            } else {
                return -961109;
            }
        }
        groupElectronegativity /= (1 + childTree.length);
        return groupElectronegativity;
    }

    //   Looking for the son group
    private String findAllChildPath(int childRoot, Set<Integer> listSet, int[][] rootedArray) {
        Set<Integer> resSet = new HashSet<>();
        resSet.add(childRoot);
        boolean flag = true;
        while (flag) {
            flag = false;
            for(int i = 0; i < rootedArray.length; i++) {
                if(listSet.contains(rootedArray[i][0]) && listSet.contains(rootedArray[i][1])) {
                    if(resSet.contains(rootedArray[i][0]) && (!resSet.contains(rootedArray[i][1]))) {
                        resSet.add(rootedArray[i][1]);
                        flag = true;
                        break;
                    } else if(resSet.contains(rootedArray[i][1]) && (!resSet.contains(rootedArray[i][0]))) {
                        resSet.add(rootedArray[i][0]);
                        flag = true;
                        break;
                    }
                }
            }
        }
        List<Integer> list = new ArrayList<>();
        for(int i : resSet) {
            list.add(i);
        }
        Collections.sort(list);
        String res = "";
        for(int i : list) {
            res += i + ",";
        }
        return res;
    }


    //    Find Root's descendants
    private int[] findRootChild(int[] childRoot, int numberOfRoot, List list, int[][] adjacencyMatrix) {
        int index = 0;
        for(int i = 0; i < adjacencyMatrix.length; i++) {
            if(adjacencyMatrix[numberOfRoot][i] == 1 && list.contains(i)) {
                childRoot[index++] = i;
            }
        }
        return childRoot;
    }

    //    Look for root in every possible situation
    private int findGroupRoot(List list, int[][] adjacencyMatrix) {
        for(Object num : list) {
            for(int i = 0; i < adjacencyMatrix.length; i++) {
                if(adjacencyMatrix[(int) num][i] == 1 && (!list.contains(i))) {
                    return (int) num;
                }
            }
        }
        return -1;
    }

    private List<List<Integer>> findAllGroup(int[][] rootedArray, List<List<Integer>> groupLists, int Cnum) {
        List<List<Integer>> allGroupList = new ArrayList<>();
        for(List list : groupLists) {
//            Find possible and complementary combinations
            Set<Integer> carbonSet = new HashSet<>();
            for(Object i : list) {
                carbonSet.add((int)i);
            }
            List<Integer> tmp = new ArrayList<>();
            for(int i = 0; i < Cnum; i++) {
                tmp.add(i);
            }
            for(int i = tmp.size() - 1; i >= 0; i--) {
                if(carbonSet.contains(tmp.get(i))) {
                    tmp.remove(i);
                }
            }
            Common common = new Common();
            if(common.connectedGraph(list, rootedArray) && common.connectedGraph(tmp, rootedArray)) {
                allGroupList.add(list);
            }

        }
        return allGroupList;
    }

    //    Add the electronegativity results to the MAP
    private double groupMapInit() {
        double electronegativity = groupEquation(GroupElectronegativity.Electronegativity.HYDROGEN.getValue(),
                GroupElectronegativity.Electronegativity.HYDROGEN.getValue(),
                GroupElectronegativity.Electronegativity.HYDROGEN.getValue());
        return electronegativity;
    }

    private double groupEquation(double value, double value1, double value2) {
        return (GroupElectronegativity.Electronegativity.CARBON.getValue() + value + value1 + value2) / 4;
    }
}
