
import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { createContext} from "react";
import Formx from './pages/Formx';
import Nextform from "./pages/Nextform";
import Nextform2 from "./pages/Nextform2";
export const Appcontext = createContext();


function App() {
  return (
    <div className="App">
      <div className="App">
        <Appcontext.Provider value={{}}>
          <Router>
            <Routes>
              <Route path="/" element={<Formx/>} />
              <Route path="/nextForm" element={<Nextform/>} />
              <Route path="/nextForm2" element={<Nextform2/>} />
            </Routes>
          </Router>
        </Appcontext.Provider>
      </div>
    </div>
  );
}

export default App;
