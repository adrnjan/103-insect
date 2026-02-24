#include <SPI.h>
#include <SD.h>

File myFile;
const int chipSelect = 5;

void loop()
{
    Serial.begin(115200);
    if (!SD.begin(chipSelect)) 
    {
    Serial.println("SD initialization failed!");
    return;
    }
  
  myFile = SD.open("/recording.wav", FILE_WRITE);
  if (myFile) 
  {
    myFile.println("Starting audio log...");
    myFile.close(); 
  }
  
}