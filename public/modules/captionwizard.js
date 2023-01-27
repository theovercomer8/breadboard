
class CaptionWizard {
    msToTime(ms) {
        let seconds = (ms / 1000).toFixed(1);
        let minutes = (ms / (1000 * 60)).toFixed(1);
        let hours = (ms / (1000 * 60 * 60)).toFixed(1);
        let days = (ms / (1000 * 60 * 60 * 24)).toFixed(1);
        if (seconds < 60) return seconds + " Sec";
        else if (minutes < 60) return minutes + " Min";
        else if (hours < 24) return hours + " Hrs";
        else return days + " Days"
    }
    constructor(app) {
        this.app = app
        if (!document.querySelector("#captionwizard-modal")) {
            let el = document.createElement('div')
            el.innerHTML = `
            
            <div class="modal micromodal-slide" id="log-modal" aria-hidden="true">
                <div class="modal__overlay" tabindex="-1" data-micromodal-close>
                    <div class="modal__container" role="dialog" aria-modal="true" aria-labelledby="log-modal-title">
                        <header class="modal__header header">
                            <h2 class="modal__title" id="log-modal-title">
                                STATUS
                            </h2>
                            <div class='flexible'></div>
                            <div id="log-modal-time"></div>
                        </header>
                        <main class="modal__content" id="log-modal-content">
                            <div id="log"></div>
                        </main>
                    </div>
                </div>
            </div>
            <div class="modal micromodal-slide" id="captionwizard-modal" aria-hidden="true">
            <div class="modal__overlay" tabindex="-1" data-micromodal-close>
              <div class="modal__container" role="dialog" aria-modal="true" aria-labelledby="captionwizard-modal-title">
                <button class="modal__close" aria-label="Close modal" data-micromodal-close><i class="fa-solid fa-x"></i></button>
                <header class="modal__header header">
                    <h2 class="modal__title" id="captionwizard-modal-title">
                        CAPTION WIZARD
                    </h2>
                    <div class='flexible'></div>
                    <div><div>Add or update captions for <span class='count'></span> images.</div><div class='operate-notice'>To operate on all visible images, please deselect images before launching caption wizard.</div>
                </header>
                <main class="modal__content" id="captionwizard-modal-content">
                  <div class='row'>
                    <div class='label'>MAX CAPTION LENGTH (0=unlimited)</div>
                    <div class='flexible'></div>
                    <input type="number" min="0" max="400" value="75" step="1" name="cap_length" />
                    <input type="range" min="0" max="400" value="75" step="1" />
                  </div>

                  <div class='row'>
                    <div class='label'>EXISTING CAPTION ACTION</div>
                    <div class='flexible'></div>
                    <input type="radio" name="existing" value="ignore" id="radio_ignore" checked/>
                    <label for="radio_ignore">IGNORE</label>
                    <input type="radio" name="existing" value="copy" id="radio_copy" />
                    <label for="radio_copy">COPY</label>
                    <input type="radio" name="existing" value="prepend" id="radio_prepend" />
                    <label for="radio_prepend">PREPEND</label>
                    <input type="radio" name="existing" value="append"  id="radio_append"/>
                    <label for="radio_append">APPEND</label>
                  </div>
                  <div class='row'>
                      <div class='label'>GIT PASS</div>
                      <div class='flexible'></div>
                      <input type="checkbox" name="git_pass" checked />
                  </div>
                  <div class='row'>
                      <div class='label'>GIT FAIL PHRASES</div>
                      <div class='flexible'></div>
                      <input type="text" name="git_fail_phrases" value="a sign that says,writing that says,that says" />
                  </div>
                  <div class='row'>
                      <div class='label'>BLIP PASS (If GIT pass enabled, will only run if GIT fail phrase encountered)</div>
                      <div class='flexible'></div>
                      <input type="checkbox" name="blip_pass" checked />
                  </div>
                  
                    <div class='row'>
                        <div class='label'>BLIP BEAMS</div>
                        <div class='flexible'></div>
                        <input type="number" min="1" max="20" value="8" step="1" name="clip_beams" />
                        <input type="range" min="1" max="20" value="8" step="1" />
                    </div>
                    <div class='row'>
                        <div class='label'>CLIP MIN LENGTH</div>
                        <div class='flexible'></div>
                        <input type="number" min="5" max="75" value="30" step="1" name="clip_min" />
                        <input type="range" min="5" max="75" value="30" step="1" />
                    </div>
                    <div class='row'>
                        <div class='label'>CLIP MAX LENGTH</div>
                        <div class='flexible'></div>
                        <input type="number" min="5" max="75" value="50" step="1" name="clip_max" />
                        <input type="range" min="5" max="75" value="50" step="1" />
                    </div>
                    <div class='row'>
                        <div class='label'>USE V2 CLIP MODEL</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="clip_v2" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>APPEND ARTIST TAGS FROM CLIP</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="clip_use_artist" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>APPEND FLAVOR TAGS FROM CLIP</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="clip_use_flavor" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>MAX FLAVORS TO APPEND</div>
                        <div class='flexible'></div>
                        <input type="number" min="1" max="10" value="4" step="1" name="clip_max_flavors" />
                        <input type="range" min="1" max="10" value="4" step="1" />
                    </div>
                    <div class='row'>
                        <div class='label'>APPEND MEDIUM TAGS FROM CLIP</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="clip_use_medium" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>APPEND MOVEMENT TAGS FROM CLIP</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="clip_use_movement" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>APPEND TRENDING TAGS FROM CLIP</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="clip_use_trending" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>TAGS TO IGNORE</div>
                        <div class='flexible'></div>
                        <input type="text" name="ignore_tags" />
                    </div>
                    <div class='row'>
                        <div class='label'>REPLACE CLASS WITH SUBJECT IN CAPTION</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="replace_class"  />
                    </div>
                    <div class='text-row'>
                        <div class='label'>SUBJECT CLASS</div>
                        <input type="text" name="sub_class" placeholder="SUBJECT CLASS TO CROP (Leave blank to auto-detect)" />
                    </div>
                    <div class='text-row'>
                        <div class='label'>SUBJECT NAME</div>
                        <input type="text" name="sub_name" placeholder="SUBJECT NAME TO REPLACE CLASS WITH IN CAPTIONS" />
                    </div>
                    <div class='row'>
                        <div class='label'>USE FOLDER AS TAG</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="folder_tag"  />
                    </div>
                    <div class='row'>
                        <div class='label'>FOLDER LEVELS TO TAG</div>
                        <div class='flexible'></div>
                        <input type="number" min="1" max="4" value="1" step="1" name="folder_tag_levels" />
                        <input type="range" min="1" max="4" value="1" step="1" />
                    </div>
                    <div class='row'>
                        <div class='label'>ENSURE UNIQUE TAGS</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="uniquify_tags" checked  />
                    </div>
                    <div class='row'>
                        <div class='label'>WRITE TO CAPTION FILE</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="write_to_file" checked />
                    </div>
                    <div class='row'>
                        <div class='label'>USE FILENAME AS EXISTING CAPTION</div>
                        <div class='flexible'></div>
                        <input type="checkbox" name="use_filename"  />
                    </div>
                </main>
                <footer class="modal__footer">
                  <button class="modal__btn modal__btn-primary">Continue</button>
                  <button class="modal__btn" data-micromodal-close aria-label="Close this dialog window">Close</button>
                </footer>
              </div>
            </div>
          </div>`
            document.body.append(el)

            MicroModal.init();
        }

        document.querySelector("#captionwizard").addEventListener("click", async (e) => {
            // e.preventDefault()
            // e.stopPropagation()
            if (this.app.selection.els.length > 0) {
                document.querySelector("#captionwizard-modal .count").innerText = `${this.app.selection.els.length}`
                document.querySelector("#captionwizard-modal .operate-notice").classList.remove("hidden")

            } else {
                document.querySelector("#captionwizard-modal .count").innerText = `${this.app.currentCount}`
                document.querySelector("#captionwizard-modal .operate-notice").classList.add("hidden")

            }
            MicroModal.show("captionwizard-modal")
        })
        document.querySelectorAll("#captionwizard-modal input[type=range]").forEach(el => el.addEventListener("input", async (e) => {
            e.preventDefault()
            e.stopPropagation()
            let box = e.target.parentElement.querySelector("input[type=number")

            box.value = e.target.value
        }))


        this.worker = new Worker("./worker.js")
        this.selected = []
        this.offset = 0
        this.total = 0
        this.progress = 0
        this.worker.onmessage = async (e) => {
            if (e.data.res.length > 0) {
                if (this.selected.length > 0) {
                    let data = e.data.res.filter((d) => {
                        return this.selected.includes(d.file_path)
                    })
                    await window.electronAPI.captionBatch(data)
                    this.offset += 1
                    await this.worker.postMessage({ query: this.app.query, sorter: this.app.navbar.sorter, offset: this.offset })
                } else {
                    this.total = e.data.count
                    await window.electronAPI.captionBatch(e.data.res)
                    this.offset += 1
                    await this.worker.postMessage({ query: this.app.query, sorter: this.app.navbar.sorter, offset: this.offset })
                }
            } else {
                await window.electronAPI.captionBatch([])

            }
        }
        document.querySelector("#captionwizard-modal .modal__btn-primary").addEventListener("click", async (e) => {
            let o = {}
            this.selected = []
            this.offset = 0
            this.progress = 0
            this.startTime = Date.now();
            if (this.app.selection.els.length > 0) {
                this.selected = this.app.selection.els.map((el) => {
                    return el.querySelector("img").getAttribute("data-src")
                })
            }
            this.total = this.selected.length

            document.querySelector("#log").innerHTML = ""

            o.git_fail_phrases = document.querySelector("#captionwizard-modal input[name=git_fail_phrases]").value
            o.git_pass = (document.querySelector("#captionwizard-modal input[name=git_pass]:checked")) ? true : false
            o.blip_pass = (document.querySelector("#captionwizard-modal input[name=blip_pass]:checked")) ? true : false
            o.cap_length = document.querySelector("#captionwizard-modal input[name=cap_length]").value
            o.existing = document.querySelector("#captionwizard-modal input[name=existing]:checked").value
            o.clip_beams = document.querySelector("#captionwizard-modal input[name=clip_beams]").value
            o.clip_min = document.querySelector("#captionwizard-modal input[name=clip_min]").value
            o.clip_max = document.querySelector("#captionwizard-modal input[name=clip_max]").value
            o.clip_v2 = (document.querySelector("#captionwizard-modal input[name=clip_v2]:checked")) ? true : false
            o.clip_use_flavor = (document.querySelector("#captionwizard-modal input[name=clip_use_flavor]:checked")) ? true : false
            o.clip_max_flavors = document.querySelector("#captionwizard-modal input[name=clip_max_flavors]").value
            o.clip_use_artist = (document.querySelector("#captionwizard-modal input[name=clip_use_artist]:checked")) ? true : false
            o.clip_use_medium = (document.querySelector("#captionwizard-modal input[name=clip_use_medium]:checked")) ? true : false
            o.clip_use_movement = (document.querySelector("#captionwizard-modal input[name=clip_use_movement]:checked")) ? true : false
            o.clip_use_trending = (document.querySelector("#captionwizard-modal input[name=clip_use_trending]:checked")) ? true : false
            o.ignore_tags = document.querySelector("#captionwizard-modal input[name=ignore_tags]").value
            o.replace_class = (document.querySelector("#captionwizard-modal input[name=replace_class]:checked")) ? true : false
            o.sub_class = document.querySelector("#captionwizard-modal input[name=sub_class]").value
            o.sub_name = document.querySelector("#captionwizard-modal input[name=sub_name]").value
            o.folder_tag = (document.querySelector("#captionwizard-modal input[name=folder_tag]:checked")) ? true : false
            o.folder_tag_levels = document.querySelector("#captionwizard-modal input[name=folder_tag_levels]").value
            o.uniquify_tags = (document.querySelector("#captionwizard-modal input[name=uniquify_tags]:checked")) ? true : false
            o.write_to_file = (document.querySelector("#captionwizard-modal input[name=write_to_file]:checked")) ? true : false
            o.use_filename = (document.querySelector("#captionwizard-modal input[name=use_filename]:checked")) ? true : false

            document.querySelector("#log").innerHTML = ""
            MicroModal.close("captionwizard-modal")
            MicroModal.show("log-modal", {
                onClose: async () => {
                    await window.electronAPI.captionAbort()
                    this.app.bar.go(100)
                }
            })


            window.electronAPI.onCaptionLog(async (_event, value) => {
                if (value === "--READY--") {
                    await this.worker.postMessage({ query: this.app.query, sorter: this.app.navbar.sorter, offset: this.offset })
                }
                if (value.startsWith('PROCESSED:')) {
                    if (this.progress == 0)
                        this.startTime = Date.now();

                    this.progress += 1
                    queueMicrotask(async () => {

                        let currentTimestampInMilliseconds = Date.now();
                        let millisecondsEllapsed = currentTimestampInMilliseconds - this.startTime;
                        if (0 < this.progress && 0 < millisecondsEllapsed) {
                            let speedLast = this.progress / millisecondsEllapsed; // bytes / sec
                            let estMSLeft = Math.round((this.total - this.progress) / speedLast);

                            document.querySelector("#log-modal-time").innerText = `${this.msToTime(estMSLeft)} remaining`
                        }
                        this.app.bar.go(100 * this.progress / this.total)
                    })

                }
                queueMicrotask(async () => {
                    let el = document.createElement('div')
                    el.classList.add('row')
                    el.innerText = value
                    let logEl = document.querySelector("#log")
                    logEl.append(el)
                    while (logEl.childElementCount > 1000)
                        logEl.removeChild(logEl.firstChild)


                })
            })
            await window.electronAPI.caption(o)
        })
    }
}