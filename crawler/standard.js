const fs = require('fs');
const path = require('path');
const { fdir } = require("fdir");
const GM = require('./gm')
const Parser = require('./parser')
const parser = new Parser()
class Standard {
  constructor(folderpath) {
    this.folderpath = folderpath
    this.gm = new GM()
  }
  async init() {
  }
  async sync(filename, force) {

    const captionsFilename = path.join(path.dirname(filename), path.basename(filename, path.extname(filename)) + '.txt');
    let captext = "";
    let jpeg;
    let info;
    try {
      jpeg = (filename.endsWith(".jpg") || filename.endsWith(".jpeg"))
      info = await this.gm.get(filename)
  
    }
    catch (e){
      console.log("E2", e)

    }
    
    try {
      captext = await fs.promises.readFile(captionsFilename, 'utf8')
    } catch (e) {
      if (e.code !== 'ENOENT') {
        throw e;
      }
    }
    try {
      // no XMP tag => parse the custom headers and convert to XMP
      if (!jpeg && (!info.parsed || force)) {
        let buf = await fs.promises.readFile(filename)
        let parsed = await parser.parse(buf)
        

        // if (!parsed.app) {
        //   // no app found => try parse from external txt file
        //   const parametersFilename = path.join(path.dirname(filename), path.basename(filename, path.extname(filename)) + '.txt');

        //   try {
        //     let parametersText = await fs.promises.readFile(parametersFilename, 'utf8')
        //     parsed = await parser.parseParametersText(parsed, parametersText);
        //   } catch (e) {
        //     if (e.code !== 'ENOENT') {
        //       throw e;
        //     }
        //   }
        // }

        let list = parser.convert(parsed)
        await this.gm.set(filename, list)
      }

      let serialized = (jpeg) ? await parser.serializeJpeg(this.folderpath, filename) : await parser.serialize(this.folderpath, filename)
      serialized.caption = captext
      serialized.has_caption = captext != ""
      return serialized
    } catch (e) {
      console.log("E", e)
    }
  }
}
module.exports = Standard
