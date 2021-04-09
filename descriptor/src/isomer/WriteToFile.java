package isomer;

import java.io.File;
import java.io.FileWriter;

public class WriteToFile {
    public void writeToFile(String data) {
        String path = "D:/file/";
        String fileName = "carbonSet2.txt";
        if(data != null) {
            try {
                File file = new File(path + fileName);
                if(!file.exists()) {
                    File dir = new File(file.getParent());
                    dir.mkdirs();
                    file.createNewFile();
                }
                FileWriter fileWriter = new FileWriter(file, true);
                fileWriter.write(data + "\n");
                fileWriter.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
