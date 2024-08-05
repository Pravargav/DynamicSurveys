import React, { useState, useEffect } from "react";
import Form from "react-bootstrap/Form";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Axios from "axios";
import Button from "react-bootstrap/Button";

function Nextform2() {
  const [checkbox1, setCheckbox1] = useState(0);
  const [checkbox2, setCheckbox2] = useState(0);
  const [checkbox3, setCheckbox3] = useState(0);
  const [checkbox4, setCheckbox4] = useState(0);

  const [select1, setSelect1] = useState(0);
  const [select2, setSelect2] = useState(0);
  const [select3, setSelect3] = useState(0);
  const [select4, setSelect4] = useState(0);

  const [integral, setIntegral] = useState(0);

  const [response, setResponse] = useState({});

  useEffect(() => {
    const addToList = async () => {
      try {
        const res = await Axios.get("http://127.0.0.1:5000/mine2", {
          params: {
            checkbox1: checkbox1,
            checkbox2: checkbox2,
            checkbox3: checkbox3,
            checkbox4: checkbox4,
            select1: select1,
            select2: select2,
            select3: select3,
            select4: select4,
            integral: integral,
          },
        });

        setResponse(res.data);
      } catch (error) {
        console.error("Error adding to list:", error);
      }
    };
    addToList();
  }, [checkbox1, checkbox2, checkbox3, checkbox4, select1, select2, select3, select4, integral]);

  const handleCheckboxChange1 = (e) => {
    if (checkbox1 === 0) {
      setCheckbox1(1);
    } else {
      setCheckbox1(0);
    }
  };
  const handleCheckboxChange2 = (e) => {
    if (checkbox2 === 0) {
      setCheckbox2(1);
    } else {
      setCheckbox2(0);
    }
  };
  const handleCheckboxChange3 = (e) => {
    if (checkbox3 === 0) {
      setCheckbox3(1);
    } else {
      setCheckbox3(0);
    }
  };
  const handleCheckboxChange4 = (e) => {
    if (checkbox4 === 0) {
      setCheckbox4(1);
    } else {
      setCheckbox4(0);
    }
  };

  const handleIntegralChange = (e) => {
    setIntegral(e.target.value);
  };

  return (
    <div>
      <h3
        style={{
          fontFamily: "sans-serif",
          textAlign: "center",
          margin: "5px 5px 5px 5px",
        }}
      >
        Test Your Problem!!
      </h3>
      <div
        style={{
          justifyContent: "center",
          alignItems: "center",
          height: "80vh",
          width: "40vh 40vh",
          border: "1px solid gray",
          margin: "20px 140px 70px 140px",
        }}
      >
        <Form>
          <Row className="justify-content-md-center">
            <Col xs lg="2">
              <Form.Check // prettier-ignore
                inline
                type="switch"
                id="Fever"
                label="Fever"
                size="lg"
                checked={checkbox1}
                onChange={handleCheckboxChange1}
                style={{
                  width: "70px", // Adjust width as needed
                  height: "70px", // Adjust height as needed
                  fontSize: "1.6rem", // Adjust font size as needed
                }}
              />
            </Col>
            <Col xs lg="2">
              <Form.Check // prettier-ignore
                inline
                type="switch"
                label="Cough"
                id="Cough"
                size="lg"
                checked={checkbox2}
                onChange={handleCheckboxChange2}
                style={{
                  width: "70px", // Adjust width as needed
                  height: "70px", // Adjust height as needed
                  fontSize: "1.6rem", // Adjust font size as needed
                }}
              />
            </Col>
          </Row>
          <Row className="justify-content-md-center">
            <Col xs lg="2">
              <Form.Check // prettier-ignore
                inline
                type="switch"
                id="Fatigue"
                label="Fatigue"
                size="lg"
                checked={checkbox3}
                onChange={handleCheckboxChange3}
                style={{
                  width: "70px", // Adjust width as needed
                  height: "70px", // Adjust height as needed
                  fontSize: "1.6rem", // Adjust font size as needed
                }}
              />
            </Col>
            <Col xs lg="2">
              <Form.Check // prettier-ignore
                inline
                type="switch"
                label="Breath"
                id="Breath"
                size="lg"
                checked={checkbox4}
                onChange={handleCheckboxChange4}
                style={{
                  width: "70px", // Adjust width as needed
                  height: "70px", // Adjust height as needed
                  fontSize: "1.6rem", // Adjust font size as needed
                }}
              />
            </Col>
          </Row>
          <Row
            className="justify-content-md-center"
            style={{ marginBottom: "1rem" }}
          >
            <Col xs={4} />
            <Col>
              <Form.Select aria-label="Default select example" size="lg" onChange={(e) => setSelect1(e.target.value)}>
                <option>Select your Gender!</option>
                <option value="1">One</option>
                <option value="2">Two</option>
              </Form.Select>
            </Col>
            <Col xs={4} />
          </Row>
          <Row
            className="justify-content-md-center"
            style={{ marginBottom: "1rem" }}
          >
            <Col xs={4} />
            <Col>
              <Form.Select aria-label="Default select example" size="lg" onChange={(e) => setSelect2(e.target.value)}>
                <option>Blood Pressure Levels!</option>
                <option value="1">One</option>
                <option value="2">Two</option>
                <option value="3">Three</option>
              </Form.Select>
            </Col>
            <Col xs={4} />
          </Row>
          <Row
            className="justify-content-md-center"
            style={{ marginBottom: "1rem" }}
          >
            <Col xs={4} />
            <Col>
              <Form.Select aria-label="Default select example" size="lg" onChange={(e) => setSelect3(e.target.value)}>
                <option>Cholestrol Levels!</option>
                <option value="1">One</option>
                <option value="2">Two</option>
                <option value="3">Three</option>
              </Form.Select>
            </Col>
            <Col xs={4} />
          </Row>
          <Row
            className="justify-content-md-center"
            style={{ marginBottom: "1rem" }}
          >
            <Col xs={4} />
            <Col>
              <Form.Select aria-label="Default select example" size="lg" onChange={(e) => setSelect4(e.target.value)}>
                <option>OutCome Variable!</option>
                <option value="1">One</option>
                <option value="2">Two</option>
              </Form.Select>
            </Col>
            <Col xs={4} />
          </Row>
          <Row
            className="justify-content-md-center"
            style={{ marginBottom: "1rem" }}
          >
            <Col xs={4} />
            <Col>
              <Form.Control
                size="lg"
                placeholder="Age"
                type="number"
                value={integral}
                onChange={handleIntegralChange}
              />
            </Col>
            <Col xs={4} />
          </Row>
        </Form>
        <Row
          className="justify-content-md-center"
          style={{ marginBottom: "1rem" }}
        >
          <Button
            type="submit"
            size="lg"
            style={{ fontFamily: "sans-serif" }}
            className="w-50"
          >
            Click {response.valx}!!
          </Button>
        </Row>
      </div>
    </div>
  );
}

export default Nextform2;
