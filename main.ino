#include <SPI.h>
#include <SD.h>

File myFile;
const int chipSelect = 5;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect.
  }

  if (!SD.begin(chipSelect)) {
    Serial.println("ERROR: SD initialization failed!");
    return;
  }
  Serial.println("INFO: SD initialized.");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("PUSH ")) {
      // Format: PUSH /filename size
      int firstSpace = command.indexOf(' ');
      int lastSpace = command.lastIndexOf(' ');
      
      if (firstSpace != -1 && lastSpace != -1 && firstSpace != lastSpace) {
        String filename = command.substring(firstSpace + 1, lastSpace);
        long fileSize = command.substring(lastSpace + 1).toInt();

        if (SD.exists(filename.c_str())) {
          SD.remove(filename.c_str());
        }

        myFile = SD.open(filename.c_str(), FILE_WRITE);
        if (myFile) {
          Serial.println("READY");
          long bytesRead = 0;
          uint32_t lastMessage = millis();
          
          while (bytesRead < fileSize) {
            if (Serial.available()) {
              byte b = Serial.read();
              myFile.write(b);
              bytesRead++;
              lastMessage = millis();
            }
            // Timeout if no data for 10 seconds
            if (millis() - lastMessage > 10000) {
              Serial.println("ERROR: Timeout");
              break;
            }
          }
          myFile.close();
          if (bytesRead == fileSize) {
            Serial.println("DONE");
          }
        } else {
          Serial.println("ERROR: Open failed");
        }
      } else {
        Serial.println("ERROR: Invalid command format");
      }
    }
  }
}