package isomer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        FileInputStream inputStream = new FileInputStream("src/isomer/graph");
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream,"utf-8"));
        Scanner in = new Scanner(System.in);
        int Cnum = in.nextInt();
        String str = null;
        List<int[][]> edgelist = new ArrayList<>();
        while ((str = bufferedReader.readLine()) != null) {
            if(str .equals("<EDGELIST>")) {
                int[][] arr = new int[Cnum - 1][2];
                int i = 0;
                while(i < Cnum - 1) {
                    str = bufferedReader.readLine();
                    String[] strr = str.split(" ");
                    arr[i][0] = Integer.parseInt(strr[0]);
                    arr[i][1] = Integer.parseInt(strr[1]);
                    i++;
                }
                edgelist.add(arr);
            }
        }

        inputStream.close();
        bufferedReader.close();

        NumberTheMainChain numberTheMainChain = new NumberTheMainChain();

//        All the subset
        Common common = new Common();
        List<List<Integer>> groupLists = common.subsets(Cnum);
//        Eliminate the impossible
        for(int i = groupLists.size() - 1; i >= 0; i--) {
            if(groupLists.get(i).size() == 0 || groupLists.get(i).size() == Cnum) {
                groupLists.remove(groupLists.get(i));
            }
        }

//        The output stream
        WriteToFile writeToFile = new WriteToFile();
        for(int j = 0; j < edgelist.size(); j++) {
            AlkaneInfo alkaneInfo = numberTheMainChain.findMainChain(edgelist.get(j), Cnum);
            int[][] adjacencyMatrix = alkaneInfo.getAdjacencyMatrix();
            int[][] rootedArray = alkaneInfo.getRootedArray();

//            Print the adjacency matrix
            System.out.println("adjacencyMatrix :" + (j + 1));
            for(int i = 0; i < adjacencyMatrix.length; i++) {
                String strr = "";
                for (int k = 0; k < adjacencyMatrix.length; k++) {
                    strr += adjacencyMatrix[i][k] + ",";
                }
                System.out.println(strr + "---------");
            }

//            Generate three adjacency matrices
            MatrixCalculate matrixCalculate = new MatrixCalculate();
            alkaneInfo = matrixCalculate.generateDistanceMatrix(adjacencyMatrix, alkaneInfo);

//            Generate all distance matrices
            DistanceMatrix distanceMatrix = new DistanceMatrix();
            int[][] distanceMat = distanceMatrix.distanceMatrix(adjacencyMatrix);
            for (int i = 0; i < distanceMat.length; i++) {
                String strrr = "";
                for (int k = 0; k < distanceMat[0].length; k++) {
                    strrr += distanceMat[i][k] + " ";
                }
            }


//            Calculate the electronegativity
            CalculateElectronegativity calculateElectronegativity = new CalculateElectronegativity();
            double[] electronegativity = calculateElectronegativity.calculateGroupElectronegativity(rootedArray, adjacencyMatrix, groupLists);
            alkaneInfo.setElectronegativity(electronegativity);

//            Calculate the degree of branching
            CalculateDegreeOfBranching calculateDegreeOfBranching = new CalculateDegreeOfBranching();
            double[] degreeOfBranching = calculateDegreeOfBranching.calculateDegree(adjacencyMatrix);
            alkaneInfo.setDegreeOfBranching(degreeOfBranching);

//            Generate an augmented matrix
            alkaneInfo = matrixCalculate.generateAugmentedMatrix(alkaneInfo);

//            Calculating topological index
            alkaneInfo = matrixCalculate.calculateIndex(alkaneInfo);
            double[] index = alkaneInfo.getIndex();
            String indexStr = "";
            for(int i = 0; i < index.length; i++) {
                indexStr += index[i] + " ";
            }
//            get rid of carbon atoms that have symmetry
            List<Integer> differentCarbonList = distanceMatrix.findSameCarbon(adjacencyMatrix, distanceMat);
            for (Integer i : differentCarbonList) {
                String outString  = indexStr + i;
            }
            System.out.println(indexStr.substring(0,indexStr.length() - 1));
        }
    }
}
