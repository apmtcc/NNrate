package isomer;

public class CalculateDegreeOfBranching {
    public double[] calculateDegree(int[][] adjacencyMatrix) {
        double[] degreeOfBranching = new double[adjacencyMatrix.length];
        for(int i = 0; i < adjacencyMatrix.length; i++) {
            for(int j = 0; j < adjacencyMatrix[0].length; j++) {
                degreeOfBranching[i] += adjacencyMatrix[i][j];
            }
            degreeOfBranching[i] = Math.sqrt(degreeOfBranching[i]);
        }

        return degreeOfBranching;
    }
}

