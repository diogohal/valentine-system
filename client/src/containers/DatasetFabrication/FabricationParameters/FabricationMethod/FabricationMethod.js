import React, {Component} from "react";
import classes from "./FabricationMethod.module.css";
import Checkbox from "@material-ui/core/Checkbox";
import {TextField} from "@material-ui/core";


class FabricationMethod extends Component{

    state = {
        selected: false,
        vInstances: true,
        nInstances: true,
        vSchemata: true,
        nSchemata: true,
        numberOfPairs: 0,
        verticalOverlapPercentage: null
    }

    componentDidMount() {
        if (this.props.methodName==="Joinable"){
            this.setState({nInstances: false})
        }else if(this.props.methodName==="Semantically Joinable"){
            this.setState({vInstances: false})
        }
    }

    supportsVerticalOverlap = () => {
        return this.props.methodName === "Joinable" ||
            this.props.methodName === "View Unionable" ||
            this.props.methodName === "Semantically Joinable"
    }

    toggleSelected = () => {
        const updatedState = {...this.state}
        updatedState.selected = !updatedState.selected
        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    toggleNoisyInstances = () => {
        if (this.props.methodName==="Joinable" || this.props.methodName==="Semantically Joinable"){
            return
        }
        const updatedState = {...this.state}
        updatedState.nInstances = !updatedState.nInstances
        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    toggleVerbatimInstances = () => {
        if (this.props.methodName==="Joinable" || this.props.methodName==="Semantically Joinable"){
            return
        }
        const updatedState = {...this.state}
        updatedState.vInstances = !updatedState.vInstances
        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    toggleNoisySchemata = () => {
        const updatedState = {...this.state}
        updatedState.nSchemata = !updatedState.nSchemata
        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    toggleVerbatimSchemata = () => {
        const updatedState = {...this.state}
        updatedState.vSchemata = !updatedState.vSchemata
        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    changeNumberOfPairs = (event) => {
        const value = parseInt(event.target.value)
        const updatedState = {...this.state}
        let valueToUpdate = 0
        if (value){
            valueToUpdate = value
        }
        updatedState.numberOfPairs = valueToUpdate
        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    changeVerticalOverlapPercentage = (event) => {
        const value = parseFloat(event.target.value)
        const updatedState = {...this.state}

        if (Number.isNaN(value)) {
            updatedState.verticalOverlapPercentage = null
        } else if (value < 0) {
            updatedState.verticalOverlapPercentage = 0
        } else if (value > 1) {
            updatedState.verticalOverlapPercentage = 1
        } else {
            updatedState.verticalOverlapPercentage = value
        }

        this.props.sendSelected(updatedState)
        this.setState(updatedState)
    }

    render() {

        const noisyInstancesJoinable = (this.props.methodName === "Joinable") ?
                    <Checkbox disabled inputProps={{ 'aria-label': 'disabled checked checkbox' }} />
                    : <Checkbox disabled checked inputProps={{ 'aria-label': 'disabled checked checkbox' }} />

        const noisyInstances = <div className={classes.Choice}>
            {(this.props.methodName !== "Joinable" && this.props.methodName !== "Semantically Joinable") ?
                <Checkbox
                    checked={this.state.nInstances}
                    onChange={() => this.toggleNoisyInstances()}
                    color="primary"
                /> :
                noisyInstancesJoinable
                }
                <p>Noisy instances</p>
            </div>;

        const verbatimInstancesJoinable = (this.props.methodName === "Joinable") ?
                    <Checkbox disabled checked inputProps={{ 'aria-label': 'disabled checked checkbox' }} />
                    : <Checkbox disabled inputProps={{ 'aria-label': 'disabled checked checkbox' }} />

        const verbatimInstances = <div className={classes.Choice}>
            {(this.props.methodName !== "Joinable" && this.props.methodName !== "Semantically Joinable") ?
                <Checkbox
                    checked={this.state.vInstances}
                    onChange={() => this.toggleVerbatimInstances()}
                    color="primary"
                /> :
                verbatimInstancesJoinable
                }
                <p>Verbatim instances</p>
            </div>;

        return(
            <div className={classes.FabricationMethod}>
                <div className={classes.Header}>
                    <Checkbox
                        checked={this.state.selected}
                        onChange={() => this.toggleSelected()}
                        color="primary"
                    />
                    <h4>{this.props.methodName}</h4>
                </div>
                <div>
                    <div className={classes.TextField}>
                        <TextField id={this.props.methodName} label="Number of pairs" onChange={this.changeNumberOfPairs} fullWidth/>
                    </div>
                    {this.supportsVerticalOverlap() ?
                        <div className={classes.TextField}>
                            <TextField
                                id={`${this.props.methodName}-vertical-overlap`}
                                label="Vertical overlap percentage"
                                type="number"
                                inputProps={{min: 0, max: 1, step: 0.01}}
                                value={this.state.verticalOverlapPercentage ?? ''}
                                onChange={this.changeVerticalOverlapPercentage}
                                fullWidth
                            />
                        </div> : null
                    }
                    <div className={classes.IncludeHeader}>
                        <h5> Include: </h5>
                    </div>
                    {noisyInstances}
                    <div className={classes.Choice}>
                        <Checkbox
                            checked={this.state.nSchemata}
                            onChange={() => this.toggleNoisySchemata()}
                            color="primary"
                        />
                        <p>Noisy schemata</p>
                    </div>
                    {verbatimInstances}
                    <div className={classes.Choice}>
                        <Checkbox
                            checked={this.state.vSchemata}
                            onChange={() => this.toggleVerbatimSchemata()}
                            color="primary"
                        />
                        <p>Verbatim schemata</p>
                    </div>
                </div>
            </div>

        );
    }


}

export default FabricationMethod;