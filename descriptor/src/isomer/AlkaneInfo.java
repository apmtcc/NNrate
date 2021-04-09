package isomer;

public class AlkaneInfo {
    private int[][] adjacencyMatrix;
    private int[][] rootedArray;
    private double[] electronegativity;
    private int[][] distanceMatrix2;
    private int[][] distanceMatrix3;
    private double[] degreeOfBranching;
    private double[][] augmentedMatrix1;
    private double[][] augmentedMatrix2;
    private double[][] augmentedMatrix3;
    private double[] index;

    public double[] getIndex() {
        return index;
    }

    public void setIndex(double[] index) {
        this.index = index;
    }

    public double[][] getAugmentedMatrix1() {
        return augmentedMatrix1;
    }

    public void setAugmentedMatrix1(double[][] augmentedMatrix1) {
        this.augmentedMatrix1 = augmentedMatrix1;
    }

    public double[][] getAugmentedMatrix2() {
        return augmentedMatrix2;
    }

    public void setAugmentedMatrix2(double[][] augmentedMatrix2) {
        this.augmentedMatrix2 = augmentedMatrix2;
    }

    public double[][] getAugmentedMatrix3() {
        return augmentedMatrix3;
    }

    public void setAugmentedMatrix3(double[][] augmentedMatrix3) {
        this.augmentedMatrix3 = augmentedMatrix3;
    }

    public double[] getDegreeOfBranching() {
        return degreeOfBranching;
    }

    public void setDegreeOfBranching(double[] degreeOfBranching) {
        this.degreeOfBranching = degreeOfBranching;
    }

    public int[][] getDistanceMatrix3() {
        return distanceMatrix3;
    }

    public void setDistanceMatrix3(int[][] distanceMatrix3) {
        this.distanceMatrix3 = distanceMatrix3;
    }

    public int[][] getDistanceMatrix2() {
        return distanceMatrix2;
    }

    public void setDistanceMatrix2(int[][] distanceMatrix2) {
        this.distanceMatrix2 = distanceMatrix2;
    }

    public double[] getElectronegativity() {
        return electronegativity;
    }

    public void setElectronegativity(double[] electronegativity) {
        this.electronegativity = electronegativity;
    }

    public int[][] getAdjacencyMatrix() {
        return adjacencyMatrix;
    }

    public void setAdjacencyMatrix(int[][] adjacencyMatrix) {
        this.adjacencyMatrix = adjacencyMatrix;
    }

    public int[][] getRootedArray() {
        return rootedArray;
    }

    public void setRootedArray(int[][] rootedArray) {
        this.rootedArray = rootedArray;
    }
}

