package isomer;

import java.util.*;

public class DistanceMatrix {
    public int[][] distanceMatrix(int[][] adjacencyMatrix) {
        int[][] distanceMatrix = new int[adjacencyMatrix.length][adjacencyMatrix[0].length];
//        Initializes the array to 1
        for (int i = 0; i < adjacencyMatrix.length; i++) {
            for (int j = 0; j < adjacencyMatrix[0].length; j++) {
                distanceMatrix[i][j] = -1;
            }
        }
//        Traverse the number
        for (int i = 0; i < adjacencyMatrix.length; i++) {
            Set<Integer> allDistanceSet = new HashSet<>();
            allDistanceSet.add(i);
            distanceMatrix[i][i] = 0;
            Set<Integer> tempSet = new HashSet<>();
            tempSet.add(i);
            int distance = 1;
            while (allDistanceSet.size() != distanceMatrix.length) {
                Set<Integer> nextSet = new HashSet<>();
                for (Integer num : tempSet) {
                    for (int j = 0; j < distanceMatrix[0].length; j++) {
                        if (adjacencyMatrix[num][j] == 1 && (!allDistanceSet.contains(j))) {
                            distanceMatrix[i][j] = distance;
                            nextSet.add(j);
                            allDistanceSet.add(j);
                        }
                    }
                }
                distance++;
                tempSet = new HashSet<>(nextSet);
            }
        }
        return distanceMatrix;
    }

    public List<Integer> findSameCarbon(int[][] adjacencyMatrix, int[][] distanceMatrix) {
        List<Integer> resList = new ArrayList<>();
        Map<Integer, Integer> differentCarbonMap = new HashMap<>();
//        Strives for the sum of squares
        for (int i = 0; i < distanceMatrix.length; i++) {
            int quadraticSum = 0;
            for (int j : distanceMatrix[i]) {
                quadraticSum += j * j;
            }
            if (!differentCarbonMap.containsValue(quadraticSum)) {
                differentCarbonMap.put(i, quadraticSum);
            }
        }
        for (Integer key : differentCarbonMap.keySet()) {
            resList.add(key);
        }
        Collections.sort(resList);
//        get rid of the carbon atoms that don't have the hydrogen atoms
        for (int i = resList.size() - 1; i >= 0; i--) {
            int num = 0;
            for (int j = 0; j < adjacencyMatrix.length; j++) {
                if(adjacencyMatrix[resList.get(i)][j] == 1) {
                    num++;
                }
            }
            if (num >= 4) {
                resList.remove(i);
            }
        }
        return resList;
    }
}
