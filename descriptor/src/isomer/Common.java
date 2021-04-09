package isomer;

import java.util.*;

public class Common {

    //    All the subset
    public List<List<Integer>> subsets(int Cnum) {
        int[] nums = new int[Cnum];
        for(int i = 0; i < Cnum; i++) {
            nums[i] = i;
        }
        List<List<Integer>> res = new ArrayList<List<Integer>>();

        backtrack(0, nums, res, new ArrayList<Integer>());
        Collections.sort(res,(o1, o2) -> {
            if (o1.size() < o2.size()) {
                return -1;
            } else {
                return 1;
            }
        });

        return res;
    }

    private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
        res.add(new ArrayList<>(tmp));
        for (int j = i; j < nums.length; j++) {
            tmp.add(nums[j]);
            backtrack(j + 1, nums, res, tmp);
            tmp.remove(tmp.size() - 1);
        }
    }

//    Determine whether it is a connected graph
    public boolean connectedGraph(List<Integer> list, int[][] rootedArray) {
        if(list.size() == 0) {
            return false;
        }
        boolean res = false;

        Set<Integer> numSet = new HashSet<>();
        Set<Integer> connectedNum = new HashSet<>();
        for(int i : list) {
            numSet.add(i);
        }
        connectedNum.add(list.get(0));
        boolean flag = true;
        while (flag) {
            flag = false;
            for(int i = 0; i < rootedArray.length; i++) {
                if(numSet.contains(rootedArray[i][0]) && numSet.contains(rootedArray[i][1]) &&
                        (((!connectedNum.contains(rootedArray[i][0])) && connectedNum.contains(rootedArray[i][1])) ||
                                (connectedNum.contains(rootedArray[i][0]) && (!connectedNum.contains(rootedArray[i][1])))) ) {
                    connectedNum.add(rootedArray[i][0]);
                    connectedNum.add(rootedArray[i][1]);
                    flag = true;
                    break;
                }

            }

        }

        if(numSet.size() == connectedNum.size()) {
            res = true;
        }
        return res;
    }
}

