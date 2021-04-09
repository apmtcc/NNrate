package isomer;

public class GroupElectronegativity {
    public enum Electronegativity {
        CARBON(1,2.55), HYDROGEN(2,2.20), METHYL(3,2.2875);

        private int index;

        private double value;

        Electronegativity(int index, double value) {
            this.index = index;
            this.value = value;
        }

        public int getIndex() {
            return index;
        }

        public double getValue() {
            return value;
        }
    }
}
