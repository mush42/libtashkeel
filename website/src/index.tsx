import { do_tashkeel } from 'libtashkeel-wasm';
import { render } from 'preact';
import { useState } from "preact/hooks";
import logo from './assets/logo.jpeg';
import './style.css';


export function App() {
    const [inputText, setInputText] = useState("");
    const [processedText, setProcessedText] = useState("");

    const handleTashkeel = () => {
        const processed = do_tashkeel(inputText);
        setProcessedText(processed);
        document.getElementById("output").focus();
    };

    return (
        <div>
            <a href="https://github.com/mush42/libtashkeel" target="_blank">
                <img src={logo} alt="A generated image of an Arabian coffee pot" height="100" width="100" style="border-radius: 50%;"/>
            </a>
            <h3>Libtashkeel: Diacritize Arabic Text </h3>
            <section>
                <Resource
                    title="GitHub"
                    description="Main repository"
                    href="https://github.com/mush42/libtashkeel"
                />
                <Resource
                    title="Libtashkeel Crate"
                    description="use from Rust"
                    href="https://crates.io/crates/libtashkeel_base"
                />
                <Resource
                    title="npm Package"
                    description="Use from JavaScript"
                    href="https://www.npmjs.com/package/libtashkeel-wasm"
                />
            </section>
            <div class="container">
                <div class="row">
                    <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    style={{ direction: "rtl" }}
                    rows={5}
                    placeholder="Enter text to diacritize..."
                    />
                </div>
                <div class="row">
                    <button onClick={handleTashkeel}>Diacritize</button>
                </div>
                <div class="row">
                    <textarea
                    id="output"
                    value={processedText}
                    style={{ direction: "rtl" }}
                    rows={5}
                    placeholder="Diacritized text..."
                    readOnly
                    />
                </div>
            </div>
        </div>
    );
}

function Resource(props) {
    return (
        <a href={props.href} target="_blank" class="resource">
            <h2>{props.title}</h2>
            <p>{props.description}</p>
        </a>
    );
}

render(<App />, document.getElementById('app'));
