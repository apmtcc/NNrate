package isomer;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.math.BigDecimal;
import java.util.HashSet;
import java.util.Set;

public class MatrixCalculate {
    //    The adjacency matrix generates a matrix of distance 2
    public AlkaneInfo generateDistanceMatrix(int[][] adjacencyMatrix, AlkaneInfo alkaneInfo) {
        int[][] distanceMatrix2 = new int[adjacencyMatrix.length][adjacencyMatrix[0].length];
        int[][] distanceMatrix3 = new int[adjacencyMatrix.length][adjacencyMatrix[0].length];
        for(int i = 0; i < adjacencyMatrix.length; i++) {
            Set<Integer> adjacencySet = new HashSet<>();
            adjacencySet.add(i);
            for(int j = 0; j < adjacencyMatrix[0].length; j++) {
                if(adjacencyMatrix[i][j] == 1) {
                    adjacencySet.add(j);
                }
            }

            Set<Integer> distanceSet2 = new HashSet<>();
            for(int num : adjacencySet) {
                for(int j = 0; j < adjacencyMatrix[0].length; j++) {
                    if(adjacencyMatrix[num][j] == 1 && (!adjacencySet.contains(j))) {
                        distanceSet2.add(j);
                    }
                }
            }

            for(int num : distanceSet2) {
                distanceMatrix2[i][num] = 2;
            }
            Set<Integer> distanceSet3 = new HashSet<>();
            for(int num : distanceSet2) {
                for(int j = 0; j < adjacencyMatrix[0].length; j++) {
                    if(adjacencyMatrix[num][j] == 1 && (!adjacencySet.contains(j)) && (!distanceSet2.contains(j))) {
                        distanceSet3.add(j);
                    }
                }
            }
            for(int num : distanceSet3) {
                distanceMatrix3[i][num] = 3;
            }
        }

        alkaneInfo.setDistanceMatrix2(distanceMatrix2);
        alkaneInfo.setDistanceMatrix3(distanceMatrix3);
        return alkaneInfo;
    }

    //    Generate an augmented matrix
    public AlkaneInfo generateAugmentedMatrix(AlkaneInfo alkaneInfo) {
        int[][] adjacencyMatrix = alkaneInfo.getAdjacencyMatrix();
        int[][] distanceMatrix2 = alkaneInfo.getDistanceMatrix2();
        int[][] distanceMatrix3 = alkaneInfo.getDistanceMatrix3();
        double[][] augmentedMatrix1 = new double[adjacencyMatrix.length][adjacencyMatrix[0].length + 2];
        double[][] augmentedMatrix2 = new double[adjacencyMatrix.length][adjacencyMatrix[0].length + 2];
        double[][] augmentedMatrix3 = new double[adjacencyMatrix.length][adjacencyMatrix[0].length + 2];
        for(int i = 0; i < augmentedMatrix1.length; i++) {
            augmentedMatrix1[i][0] = alkaneInfo.getDegreeOfBranching()[i];
            augmentedMatrix2[i][0] = alkaneInfo.getDegreeOfBranching()[i];
            augmentedMatrix3[i][0] = alkaneInfo.getDegreeOfBranching()[i];

            augmentedMatrix1[i][1] = alkaneInfo.getElectronegativity()[i];
            augmentedMatrix2[i][1] = alkaneInfo.getElectronegativity()[i];
            augmentedMatrix3[i][1] = alkaneInfo.getElectronegativity()[i];
            for(int j = 2; j < augmentedMatrix1[0].length; j++) {
                augmentedMatrix1[i][j] = adjacencyMatrix[i][j - 2];
                augmentedMatrix2[i][j] = distanceMatrix2[i][j - 2];
                augmentedMatrix3[i][j] = distanceMatrix3[i][j - 2];
            }
        }
        alkaneInfo.setAugmentedMatrix1(augmentedMatrix1);
        alkaneInfo.setAugmentedMatrix2(augmentedMatrix2);
        alkaneInfo.setAugmentedMatrix3(augmentedMatrix3);

        return alkaneInfo;
    }

    public AlkaneInfo calculateIndex(AlkaneInfo alkaneInfo) {
        double[][] augmentedMatrix1 = alkaneInfo.getAugmentedMatrix1();
        double[][] augmentedMatrix2 = alkaneInfo.getAugmentedMatrix2();
        double[][] augmentedMatrix3 = alkaneInfo.getAugmentedMatrix3();
        double index1 = calculateMaxEigen(augmentedMatrix1);
        double index2 = calculateMaxEigen(augmentedMatrix2);
        double index3 = calculateMaxEigen(augmentedMatrix3);
        double[] index = {index1, index2, index3};
        alkaneInfo.setIndex(index);

        return alkaneInfo;
    }

    private double calculateMaxEigen(double[][] augmentedMatrix) {
        for(int i = 0; i < augmentedMatrix.length; i++) {
            for(int j = 0; j < augmentedMatrix[0].length; j++) {
                BigDecimal bg = new BigDecimal(augmentedMatrix[i][j]);
                augmentedMatrix[i][j] = bg.setScale(4,BigDecimal.ROUND_HALF_UP).doubleValue();
            }
        }
        Matrix augmentedMatrix1 = new Matrix(augmentedMatrix);
        Matrix augmentedMatrix1Trans = augmentedMatrix1.transpose();
        Matrix QMatrix1 = augmentedMatrix1.times(augmentedMatrix1Trans);
        EigenvalueDecomposition eigenvalueDecomposition = new EigenvalueDecomposition(QMatrix1);
        double[] eigenValues = eigenvalueDecomposition.getRealEigenvalues();
        double maxEigenValues = Integer.MIN_VALUE;
        for(int i = 0; i < eigenValues.length; i++) {
            maxEigenValues = maxEigenValues > eigenValues[i] ? maxEigenValues : eigenValues[i];
        }
        BigDecimal bg = new BigDecimal(maxEigenValues);
        maxEigenValues = bg.setScale(4, BigDecimal.ROUND_HALF_UP).doubleValue();
        return maxEigenValues;
    }
}

