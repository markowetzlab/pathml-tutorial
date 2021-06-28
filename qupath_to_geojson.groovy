// Script to export QuPath annotations to a GeoJSON file

// Get annotation objects
def annotations = getAnnotationObjects()
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)

// Print GeoJSON (optional)
println gson.toJson(annotations)

// Where to save GeoJSON file (modify)
outputPath = "/Users/berman01/pathml-tutorial/qupath_annotations/tumor_001.json"

// Save GeoJSON to a file
File file = new File(outputPath)
file.withWriter('UTF-8') 
{
    gson.toJson(annotations, it)
}